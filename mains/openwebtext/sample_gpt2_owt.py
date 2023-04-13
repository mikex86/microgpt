import torch

from inference.sampler import AutoregressiveSampler
from models.gpt2 import Gpt2Config, Gpt2Model, Gpt2Tokenizer
from train import checkpointing


def main():
    # device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpt_config = Gpt2Config(
        block_size=1024,
        n_layers=12,
        n_heads=12,
        n_embd=768,
        device=device,
        dtype=torch.float32,
    )
    model = Gpt2Model(gpt_config).to(device)
    checkpointing.load_checkpoint(model, None, 'checkpoints/owt/gpt2_checkpoints', 'best')

    tokenizer = Gpt2Tokenizer()
    sampler = AutoregressiveSampler(model, tokenizer)

    prompt = "The Java Spring Framework"
    sampler.stream_text(
        prompt,
        # gpt2config.block_size - tokenizer.get_num_tokens(prompt),
        256 * 4,
        lambda token_str: print(token_str, end='')
    )


if __name__ == '__main__':
    main()
