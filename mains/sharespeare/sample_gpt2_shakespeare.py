import torch

from inference.sampler import AutoregressiveSampler
from models.gpt2 import Gpt2Config, Gpt2Model
from tokenization.greedy_tokenizer import GreedyTokenizer
from train import checkpointing


def main():
    # device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpt2config = Gpt2Config(
        vocab_size=65,
        block_size=256*4,
        n_layers=6,
        n_heads=6,
        n_embd=384
    )
    model = Gpt2Model(gpt2config).to(device)
    checkpointing.load_checkpoint(model, None, 'checkpoints/shakespeare/gpt2_checkpoints', 'best')

    tokenizer = GreedyTokenizer.from_json("datasets/shakespeare_char/vocabulary.json")
    sampler = AutoregressiveSampler(model, tokenizer)

    prompt = "\n"
    sampler.stream_text(
        prompt,
        # gpt2config.block_size - tokenizer.get_num_tokens(prompt),
        256 * 4,
        lambda token_str: print(token_str, end='')
    )


if __name__ == '__main__':
    main()
