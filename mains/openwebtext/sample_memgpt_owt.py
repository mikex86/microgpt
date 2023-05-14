import sys

import torch

from inference.sampler import AutoregressiveSampler
from models.gpt2 import Gpt2Tokenizer
from models.memgpt import MemGptConfig, MemGptModel
from train import checkpointing


def main():
    # device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    memgpt_config = MemGptConfig(
        block_size=512,
        n_windows=8,
        n_layers=12,
        n_heads=12,
        n_embd=768,
        device=device,
        dtype=torch.float32,
    )
    model = MemGptModel(memgpt_config).to(device)
    checkpointing.load_checkpoint(model, None, 'checkpoints/owt/memgpt_checkpoints', 'best')

    tokenizer = Gpt2Tokenizer()
    sampler = AutoregressiveSampler(model, tokenizer, token_blacklist=[tokenizer.tokenizer.eot_token])

    prompt = open("prompt.txt", "r").read()

    token_counter = tokenizer.get_num_tokens(prompt)

    def handle_text(new_str: str):
        nonlocal token_counter
        print(new_str, end='', flush=True)
        token_counter += 1
        if token_counter == memgpt_config.block_size:
            sys.stderr.write('\n<|block_end|>\n')
            sys.stderr.flush()
            token_counter = 0

    sampler.stream_text(
        prompt,
        (memgpt_config.block_size * memgpt_config.n_windows) - tokenizer.get_num_tokens(prompt),
        lambda text: handle_text(text),
    )


if __name__ == '__main__':
    main()
