import torch

from inference.sampler import AutoregressiveSampler
from models.memgpt import MemGptConfig, MemGptModel
from tokenization.greedy_tokenizer import GreedyTokenizer
from train import checkpointing


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    mem_gpt_config = MemGptConfig(
        vocab_size=65,
        block_size=256,
        n_windows=4,
        n_layers=6,
        n_heads=6,
        n_embd=384
    )
    model = MemGptModel(mem_gpt_config).to(device)
    checkpointing.load_checkpoint(model, None, 'checkpoints/shakespeare/memgpt_checkpoints', 'best')

    tokenizer = GreedyTokenizer.from_json("datasets/shakespeare_char/vocabulary.json")
    sampler = AutoregressiveSampler(model, tokenizer)

    prompt = "ROMEO:\n"
    print(prompt, end='')
    sampler.stream_text(
        prompt,
        mem_gpt_config.block_size * mem_gpt_config.n_windows - len(prompt),
        lambda token: print(token, end='')
    )


if __name__ == '__main__':
    main()
