import multiprocessing
import os
import time
from typing import Optional, Mapping, Dict

import numpy as np
import s3fs as s3fs
import torch
from tqdm import tqdm

from models.moduleapi import ILanguageModel
from tokenization.tokenizer import Tokenizer, TerminatedTokenizer
from utils.iterator_utils import make_batched_iterator, prefetching_iterator


def get_optimal_batch_size(model: ILanguageModel, tokenizer: Tokenizer, device: torch.device, block_size: int) -> int:
    batch_size = 1
    print("Determining optimal batch size...")
    batch_list = []

    n_trials = 5

    highest_throughput_batch_size = 0
    prev_highest_throughput = 0

    while True:
        block = torch.randint(0, tokenizer.vocab_size, (block_size,), dtype=torch.long, device=device)
        batch_list.append(block)

        batch = torch.stack(batch_list, dim=0)

        try:
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                start.record()
            else:
                start = time.time()

            for _ in range(n_trials):
                model(batch)

            if device.type == "cuda":
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                torch.cuda.synchronize()
                time_elapsed = start.elapsed_time(end)
            else:
                end = time.time()
                time_elapsed = (end - start) * 1000

            time_elapsed /= n_trials

            n_tokens = batch_size * block_size
            throughput = n_tokens / (time_elapsed / 1000)
            if throughput > prev_highest_throughput:
                highest_throughput_batch_size = batch_size
                prev_highest_throughput = throughput

            print("Batch size: {}, time: {} ms, throughput: {} tokens/s".format(batch_size, time_elapsed,
                                                                                throughput))
        except RuntimeError as e:
            if "out of memory" in str(e):

                while len(batch_list) > 0:
                    del batch_list[0]

                del batch

                torch.cuda.empty_cache()

                print(f"Highest throughput batch size is: {highest_throughput_batch_size}")
                return highest_throughput_batch_size
            else:
                raise e
        batch_size += 1


s3: Optional[s3fs.S3FileSystem] = None
files = []

BUFFER_SIZE = 65536

N_SEQUENTIAL_BLOCKS = 10
SEQUENTIAL_SKIP = 1000


class BlockStreamingProcess(multiprocessing.Process):

    def __init__(self, file: str, token_dtype: np.dtype, block_size: int, queue: multiprocessing.Queue):
        super().__init__()
        self.file = s3.open(file, 'rb')
        self.token_dtype = token_dtype
        self.block_size = block_size
        self.queue = queue

    def read_next_block(self) -> np.ndarray:
        dtype_bytes = np.dtype(self.token_dtype).itemsize
        block = self.file.read(self.block_size * dtype_bytes)
        block = np.frombuffer(block, dtype=self.token_dtype).astype(np.int32)
        return block

    def run(self) -> None:
        while True:
            _ = self.queue.get()  # wait for a signal to start reading
            block = self.read_next_block()
            self.queue.put(block)


def parallel_block_iterator(dataset_s3_folder: str, block_size: int, is_train: bool,
                            token_dtype: np.dtype, num_workers: int, blocks_in_flight: int) -> iter:
    results = []
    first_result_yielded = False

    # Create block streaming processes for all files
    

    while True:

        # yield the first result
        for result in results:
            if result.ready():
                block = result.get()
                yield block
                results.remove(result)
                first_result_yielded = True
                break
        # else:
        # if first_result_yielded:
        # we are starved for data
        # print("Starved for data...")


@torch.inference_mode()
def logitify_targets(model: ILanguageModel, tokenizer: Tokenizer,
                     block_size: int, batch_size: int, token_budget: int,
                     device: torch.device,
                     dataset_s3_folder: str,
                     token_dtype: np.dtype):
    try:
        torch.compile(model)
    except RuntimeError:
        print("Model compilation not supported, skipping...")

    model.eval()

    num_workers = 32
    blocks_in_flight = 64

    num_blocks = token_budget // block_size

    # train_it = block_iterator(dataset_s3_folder, block_size, True, tokenizer, token_dtype)
    train_it = parallel_block_iterator(dataset_s3_folder, block_size, True, token_dtype,
                                       num_workers,
                                       blocks_in_flight)

    batch = []
    with tqdm(desc="Logitifying targets", total=num_blocks * block_size, unit="tokens") as pbar:
        for block in train_it:
            pbar.update(block_size)
            pass
            # batch.append(block)
            # if len(batch) == batch_size:
            #     batch = np.stack(batch, axis=0)
            #     batch = torch.from_numpy(batch).pin_memory().to(device, non_blocking=True)
            #
            #     logits = model(batch)
            #
            #     del logits
            #     del batch
            #
            #     batch = []
            #
            #     pbar.update(batch_size * block_size)
