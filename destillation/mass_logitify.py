import time

import numpy as np
import torch

from models.moduleapi import ILanguageModel
from tokenization.tokenizer import Tokenizer


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


@torch.inference_mode()
def logitify_targets(model: ILanguageModel, tokenizer: Tokenizer,
                     block_size: int, device: torch.device,
                     targets_file_path: str,
                     token_dtype: np.dtype):
    try:
        torch.compile(model)
    except RuntimeError:
        print("Model compilation not supported, skipping...")

    model.eval()

    # array = np.memmap(targets_file_path, mode="r", dtype=token_dtype)
    # num_tokens = len(array)
    # num_blocks = num_tokens // block_size

    batch_size = get_optimal_batch_size(model, tokenizer, device, block_size)

    # batch = []
    #
    # for i in range(num_blocks):
    #     block = array[i * block_size:(i + 1) * block_size]
    #     batch.append(block.astype(np.int32))
    #
    #     if len(batch) == batch_size:
    #         batch = np.stack(batch, axis=0)
    #         batch = torch.from_numpy(batch).pin_memory().to(device, non_blocking=True)
    #         logits = model(batch)
    #
    #         del logits
    #         del batch
    #
    #         batch = []
