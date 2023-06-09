import json
import multiprocessing
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from queue import Empty
from typing import List

import numpy as np
import requests
import s3fs as s3fs
from datasets.data_files import DataFilesList
from huggingface_hub import HfApi

from robustdatasets.parquet_wrappers import FileWrappingParquetIterator
from tokenization.sentencepiece_tokenizer import SentencePieceTokenizer

from rich.progress import Table, Progress, BarColumn, TimeRemainingColumn, MofNCompleteColumn, \
    TaskProgressColumn
from rich.panel import Panel
from rich.live import Live

language_importance = json.load(open("language_importance.json", "r"))


def list_parquet_files(repo_id: str, patterns: List[str]):
    api = HfApi()
    ds_info = api.dataset_info(repo_id)
    files_list = DataFilesList.from_hf_repo(patterns, ds_info)
    return files_list


TOKEN_BUFFER_SIZE = 100000
TOKENIZE_BATCH_SIZE = 1024


def _flush_token_buffer(token_buffer, out_file):
    token_np_array = np.array(token_buffer, dtype=np.uint16)
    token_bytes = token_np_array.tobytes()
    out_file.write(token_bytes)
    out_file.flush()
    token_buffer.clear()


class OsFs:
    @staticmethod
    def open(path, mode):
        return open(path, mode)

    @staticmethod
    def exists(path):
        return os.path.exists(path)

    @staticmethod
    def touch(path, create_parents=True):
        if create_parents:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, 'a').close()


upload_to_s3 = False
if upload_to_s3:
    fs = s3fs.S3FileSystem(key=os.environ['AWS_ACCESS_KEY_ID'], secret=os.environ['AWS_SECRET_ACCESS_KEY'])
else:
    fs = OsFs()

s3_bucket = 'micro-gpt-datasets-us'
s3_prefix = 'the-stack-replit'


class Message:
    pass


@dataclass
class CreateProgressBarMessage(Message):
    task_id: str
    n_total: int


@dataclass
class SetProgressMessage(Message):
    task_id: str
    n_current_progress: int


@dataclass
class DestroyProgressBarMessage(Message):
    task_id: str


class ErrorMessage(Message):

    def __init__(self, task_id: str, exception: BaseException):
        self.task_id = task_id
        self.exception = exception
        self.traceback = traceback.extract_tb(exception.__traceback__)


def process_parquet_url(parquet_url: str, progress_queue: multiprocessing.Queue):
    lang_name = parquet_url.split("/")[-2]
    index = int(parquet_url.split("-")[-3])

    task_id = f"{lang_name}-{index}"
    try:
        train_file_path = f"train-{lang_name}-{index}.bin"
        val_file_path = f"val-{lang_name}-{index}.bin"

        train_s3_key = f"{s3_bucket}/{s3_prefix}/{train_file_path}"
        val_s3_key = f"{s3_bucket}/{s3_prefix}/{val_file_path}"

        if fs.exists(train_s3_key) and fs.exists(val_s3_key):
            return

        tokenizer = SentencePieceTokenizer("replit_tokenizer.model")

        response = requests.get(parquet_url,
                                headers={'Authorization': 'Bearer ' + os.environ['HUGGINGFACE_TOKEN']},
                                stream=True)
        response.raise_for_status()

        file_size = int(response.headers.get('Content-Length', 0))

        download_task_id = f"{task_id}-download"
        progress_queue.put(CreateProgressBarMessage(download_task_id, file_size))
        try:
            buffer = BytesIO()

            for chunk in response.iter_content(chunk_size=33554432):
                progress_queue.put(SetProgressMessage(download_task_id, buffer.tell()))
                buffer.write(chunk)

            buffer.seek(0)
        except BaseException as e:
            progress_queue.put(ErrorMessage(download_task_id, e))
            return
        finally:
            progress_queue.put(DestroyProgressBarMessage(download_task_id))

        streamer = FileWrappingParquetIterator(buffer, observed_columns=['content'])
        n_rows = len(streamer)

        progress_queue.put(CreateProgressBarMessage(task_id, n_rows))

        train_token_buffer = []
        val_token_buffer = []

        fs.touch(train_s3_key, create_parents=True)
        fs.touch(val_s3_key, create_parents=True)

        with fs.open(train_s3_key, "wb") as train_file:
            with fs.open(val_s3_key, "wb") as val_file:
                row_idx = 0

                batch = []
                for row in streamer:
                    content = row['content']
                    batch.append(content)
                    row_idx += 1

                    if len(batch) < TOKENIZE_BATCH_SIZE and row_idx != n_rows:
                        continue

                    tokens_batch = tokenizer.encode_batch(batch, eos=True)
                    batch = []

                    for tokens in tokens_batch:
                        goes_to_val = np.random.random() < 0.01
                        if goes_to_val:
                            val_token_buffer.extend(tokens)
                            if len(val_token_buffer) >= TOKEN_BUFFER_SIZE:
                                _flush_token_buffer(val_token_buffer, val_file)
                                # update progress bar
                                progress_queue.put(SetProgressMessage(task_id, row_idx))
                        else:
                            train_token_buffer.extend(tokens)
                            if len(train_token_buffer) >= TOKEN_BUFFER_SIZE:
                                _flush_token_buffer(train_token_buffer, train_file)
                                # update progress bar
                                progress_queue.put(SetProgressMessage(task_id, row_idx))

                _flush_token_buffer(train_token_buffer, train_file)
                _flush_token_buffer(val_token_buffer, val_file)

        print(f"Finished processing {train_file_path}")
        return train_file_path, val_file_path
    except Exception as e:
        print(f"Error processing {parquet_url}: {e}")
        progress_queue.put(ErrorMessage(task_id, e))
    finally:
        progress_queue.put(DestroyProgressBarMessage(task_id))


error_file = None


def print_err(string: str):
    global error_file
    if error_file is None:
        error_file = open("errors.txt", "a+")

    error_file.write(string + "\n")
    error_file.flush()


def main():
    multiprocessing.set_start_method("spawn", force=True)

    parquet_urls = list_parquet_files("bigcode/the-stack-dedup", patterns=["**/data-*-of-*.parquet"])

    # remove languages that are not important
    parquet_urls = list(filter(lambda x: language_importance.get(x.split("/")[-2], 0) > 0, parquet_urls))

    print(f"Downloading {len(parquet_urls)} parquet files")

    # multiprocessing
    #num_workers = multiprocessing.cpu_count()
    num_workers = 1

    tasks = {}

    results = []

    # Used by subprocesses to communicate progress to the main process.
    # Every process has its own queue.
    progress_queues = []

    overall_progress = Progress(
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        TimeRemainingColumn(),
        MofNCompleteColumn(),
    )
    overall_task = overall_progress.add_task("All Jobs", total=len(parquet_urls))

    jobs_progress = Progress(
        "{task.description}",
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        TimeRemainingColumn(),
        MofNCompleteColumn(),
    )

    progress_table = Table.grid()
    progress_table.add_row(
        Panel.fit(jobs_progress, title="[b]Jobs", border_style="red", padding=(1, 1))
    )
    progress_table.add_row(
        Panel.fit(
            overall_progress, title="Overall Progress", border_style="green", padding=(1, 1)
        )
    )

    with ProcessPoolExecutor(num_workers) as executor:
        for parquet_url in parquet_urls:
            progress_queue = multiprocessing.Queue()
            progress_queues.append(progress_queue)
            results.append(executor.submit(process_parquet_url, parquet_url, progress_queue))

        with Live(progress_table, refresh_per_second=10):
            while True:
                time.sleep(0.25)
                for progress_queue in progress_queues:
                    try:
                        new_message = progress_queue.get(block=False)
                    except Empty:
                        new_message = None

                    if new_message is not None:
                        if isinstance(new_message, CreateProgressBarMessage):
                            task = jobs_progress.add_task(new_message.task_id, total=new_message.n_total)
                            tasks[new_message.task_id] = task

                        elif isinstance(new_message, SetProgressMessage):
                            task = tasks.get(new_message.task_id, None)
                            if task is not None:
                                jobs_progress.update(task, completed=new_message.n_current_progress)
                            else:
                                print_err(
                                    f"Error: Received progress message for unknown task {new_message.task_id}")

                        elif isinstance(new_message, DestroyProgressBarMessage):
                            overall_progress.advance(overall_task, 1)
                            task = tasks.get(new_message.task_id, None)
                            if task is not None:
                                jobs_progress.remove_task(task)
                                del tasks[new_message.task_id]
                            else:
                                print_err(
                                    f"Error: Received destroy message for unknown task {new_message.task_id}")

                        elif isinstance(new_message, ErrorMessage):
                            task = tasks.get(new_message.task_id, None)
                            if task is not None:
                                jobs_progress.remove_task(task)
                                del tasks[new_message.task_id]
                                print_err(f"Error from task ${new_message.task_id}: {new_message.exception}")
                            else:
                                print_err(
                                    f"Error: from unknown task {new_message.task_id}: {new_message.exception}")

                            # print stacktrace
                            print_err('\n'.join(new_message.traceback.format()) + '\n')

                    # break if all tasks are done
                    all_done = all(result.done() for result in results)
                    if all_done:
                        break


if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        print_err(f"Exception in main() method: {e}")
        print_err(traceback.format_exc())
