import os
from typing import Tuple

import torch
from fairscale.nn.model_parallel import initialize_model_parallel

from inference.sampler import AutoregressiveSampler
from models.llama import LlamaTokenizer, LlamaModel, LlamaConfig
from train import checkpointing


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    # device = torch.device('cpu')

    tokenizer = LlamaTokenizer(model_path='checkpoints/llama/tokenizer.model')

    local_rank, world_size = setup_model_parallel(target_device=device)

    config = LlamaConfig(
        dim=512,
        multiple_of=256,
        n_heads=8,
        n_layers=8,
        norm_eps=1e-06,
        max_seq_len=1024,
        vocab_size=tokenizer.vocab_size,
        init_weights=False,
    )
    model = LlamaModel(config, device)
    checkpointing.load_checkpoint(model, None, 'checkpoints/shakespeare/llama_checkpoints', 'best')

    sampler = AutoregressiveSampler(model, tokenizer)
    prompt = "First Citizen:"
    print(prompt, end='')
    sampler.stream_text(
        prompt,
        256,
        lambda token_str: print(token_str, end='')
    )


if __name__ == '__main__':
    main()
