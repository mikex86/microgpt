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

file_cache: Dict[str, s3fs.S3File] = {}


def read_next_block_part(dataset_s3_folder: str, block_size: int, is_train: bool,
                         tokenizer: Tokenizer,
                         dtype: np.dtype) -> np.ndarray:
    global s3, files
    if s3 is None:
        s3 = s3fs.S3FileSystem(key=os.environ['AWS_ACCESS_KEY_ID'], secret=os.environ['AWS_SECRET_ACCESS_KEY'])
        files = s3.ls(dataset_s3_folder)

    train_files = [f for f in files if 'train' in f]
    test_files = [f for f in files if 'val' in f]

    while True:
        if is_train:
            file = np.random.choice(train_files)
        else:
            file = np.random.choice(test_files)

        file_size = s3.info(file)['Size']
        if file_size >= block_size:
            break

    dtype_bytes = np.dtype(dtype).itemsize

    f = file_cache.get(file, None)

    if f is None:
        f = s3.open(file, 'rb')
        file_cache[file] = f

    rand_idx = np.random.randint(0, file_size - block_size)

    # check dtype alignment
    if rand_idx % dtype_bytes != 0:
        rand_idx -= 1

    f.seek(rand_idx)

    if isinstance(tokenizer, TerminatedTokenizer):
        # read until next eot token
        block = f.read(block_size * dtype_bytes)
        block = np.frombuffer(block, dtype=dtype)

        # crop after first eot token
        eot_idx = np.where(block == tokenizer.eot_token)[0]
        if len(eot_idx) > 0:
            block = block[:eot_idx[0]]

    else:
        block = f.read(block_size * dtype_bytes)
        block = np.frombuffer(block, dtype=dtype)

    return block


def build_full_block(dataset_s3_folder: str, block_size: int, is_train: bool, tokenizer: Tokenizer,
                     token_dtype: np.dtype):
    block = read_next_block_part(dataset_s3_folder, block_size, is_train, tokenizer, token_dtype)
    while len(block) < block_size:
        block = np.concatenate(
            (block, read_next_block_part(dataset_s3_folder, block_size, is_train,
                                         tokenizer, token_dtype))
        )
    block = block[:block_size]
    return block.astype(np.int32)


def block_iterator(dataset_s3_folder: str, block_size: int, is_train: bool, tokenizer: Tokenizer,
                   token_dtype: np.dtype):
    while True:
        yield build_full_block(dataset_s3_folder, block_size, is_train, tokenizer, token_dtype)


def parallel_block_iterator(dataset_s3_folder: str, block_size: int, is_train: bool, tokenizer: Tokenizer,
                            token_dtype: np.dtype, num_workers: int, blocks_in_flight: int) -> iter:
    pool = multiprocessing.get_context('spawn').Pool(num_workers)

    results = []
    first_result_yielded = False

    while True:
        while len(results) < blocks_in_flight:
            # run build_full_block in parallel
            results.append(
                pool.apply_async(build_full_block, (dataset_s3_folder, block_size, is_train, tokenizer, token_dtype))
            )

        # yield the first result
        for block in results:
            if block.ready():
                yield block.get()
                results.remove(block)
                first_result_yielded = True
                break
        # else:
            # if first_result_yielded:
                # we are starved for data
                #print("Starved for data...")


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

    # train_it = prefetching_iterator(
    #     block_iterator(dataset_s3_folder, block_size, True, tokenizer, token_dtype),
    #     num_prefetch=batch_size * num_prefetch_batches
    # )
    train_it = parallel_block_iterator(dataset_s3_folder, block_size, True, tokenizer, token_dtype,
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
