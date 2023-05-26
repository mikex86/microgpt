import json
import multiprocessing
import os
from typing import List

import numpy as np
import s3fs as s3fs
from tqdm import tqdm

from robustdatasets.parquet_streamer import ParquetStreamer
from tokenization.sentencepiece_tokenizer import SentencePieceTokenizer
from huggingface_hub import HfApi

from datasets.data_files import DataFilesList

language_importance = json.load(open("language_importance.json", "r"))


def list_parquet_files(repo_id: str, patterns: List[str]):
    api = HfApi()
    ds_info = api.dataset_info(repo_id)
    files_list = DataFilesList.from_hf_repo(patterns, ds_info)
    return files_list


TOKEN_BUFFER_SIZE = 10000


def _flush_token_buffer(token_buffer, out_file):
    token_np_array = np.array(token_buffer, dtype=np.uint16)
    token_bytes = token_np_array.tobytes()
    out_file.write(token_bytes)
    out_file.flush()
    token_buffer.clear()


s3 = s3fs.S3FileSystem(key=os.environ['AWS_ACCESS_KEY_ID'], secret=os.environ['AWS_SECRET_ACCESS_KEY'])

s3_bucket = 'micro-gpt-datasets-us'
s3_prefix = 'the-stack-replit'


def process_parquet_url(parquet_url: str):
    lang_name = parquet_url.split("/")[-2]
    index = int(parquet_url.split("-")[-1].split(".")[0])

    train_file_path = f"train-{lang_name}-{index}.bin"
    val_file_path = f"val-{lang_name}-{index}.bin"

    train_s3_key = f"{s3_bucket}/{s3_prefix}/{train_file_path}"
    val_s3_key = f"{s3_bucket}/{s3_prefix}/{val_file_path}"

    if s3.exists(train_s3_key) and s3.exists(val_s3_key):
        print(f"Skipping {train_file_path} because it already exists")
        return

    tokenizer = SentencePieceTokenizer("replit_tokenizer.model")
    streamer = ParquetStreamer(
        parquet_url,
        headers={'Authorization': 'Bearer ' + os.environ['HUGGINGFACE_TOKEN']},
        observed_rows=['content']
    )

    train_token_buffer = []
    val_token_buffer = []

    s3.touch(train_s3_key, create_parents=True)
    s3.touch(val_s3_key, create_parents=True)
    with s3.open(train_s3_key, "wb") as train_file:
        with s3.open(val_s3_key, "wb") as val_file:
            for row in streamer:
                goes_to_val = np.random.random() < 0.01
                content = row['content']
                tokens = tokenizer.encode(content, eos=True)

                if goes_to_val:
                    val_token_buffer.extend(tokens)
                    if len(val_token_buffer) >= TOKEN_BUFFER_SIZE:
                        _flush_token_buffer(val_token_buffer, val_file)
                else:
                    train_token_buffer.extend(tokens)
                    if len(train_token_buffer) >= TOKEN_BUFFER_SIZE:
                        _flush_token_buffer(train_token_buffer, train_file)

            _flush_token_buffer(train_token_buffer, train_file)
            _flush_token_buffer(val_token_buffer, val_file)

    return train_file_path, val_file_path


BUFFER_SIZE = 65536 * 16


def main():
    parquet_urls = list_parquet_files("bigcode/the-stack-dedup", patterns=["**/data-00000-of-*.parquet"])

    # remove languages that are not important
    parquet_urls = list(filter(lambda x: language_importance.get(x.split("/")[-2], 0) > 0, parquet_urls))

    # multiprocessing
    num_workers = multiprocessing.cpu_count()
    results = []
    with multiprocessing.get_context('spawn').Pool(num_workers) as pool:
        with tqdm(total=len(parquet_urls), desc="Downloading bigcode/the-stack", unit="parquet files") as pbar:
            def update_progress():
                pbar.update()

            result = pool.map_async(process_parquet_url, parquet_urls, callback=update_progress)
            results.append(result)

        # wait for all the results to finish
        for result in results:
            result.wait()

        # check for errors
        for result in results:
            if result.successful():
                continue
            else:
                print("Error in multiprocessing")
                print(result.get())


if __name__ == '__main__':
    main()
