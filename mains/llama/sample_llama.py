import os
from typing import Tuple

import torch
from fairscale.nn.model_parallel import initialize_model_parallel

from inference.sampler import AutoregressiveSampler
from models.llama import LlamaTokenizer, LlamaModel


def setup_model_parallel(target_device: torch.device) -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    is_cuda = target_device.type == 'cuda'

    if is_cuda and os.name != 'nt':
        torch.distributed.init_process_group("nccl")
    else:
        torch.distributed.init_process_group("gloo")

    initialize_model_parallel(world_size)

    if is_cuda:
        torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def main():
    # chose between cpu, cuda and mps
    # device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device('cpu')

    tokenizer = LlamaTokenizer(model_path='checkpoints/llama/tokenizer.model')

    local_rank, world_size = setup_model_parallel(target_device=device)

    llama_model = LlamaModel.load('checkpoints/llama/7B',
                                  tokenizer=tokenizer,
                                  target_device=device,
                                  local_rank=local_rank, world_size=world_size)

    sampler = AutoregressiveSampler(llama_model, tokenizer)
    prompt = "The Java Spring Framework"
    print(prompt, end='')
    sampler.stream_text(
        prompt,
        256,
        lambda token_str: print(token_str, end='')
    )


if __name__ == '__main__':
    main()
