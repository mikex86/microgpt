from inference.sampler import AutoregressiveSampler
from models.gpt2 import Gpt2Config, Gpt2Model
from tokenization.greedy_tokenizer import GreedyTokenizer
from train import checkpointing


def main():
    gpt2config = Gpt2Config(
        vocab_size=65,
        block_size=256,
        n_layers=6,
        n_heads=6,
        n_embd=384
    )
    model = Gpt2Model(gpt2config)
    checkpointing.load_checkpoint(model, None, 'checkpoints', 'best')

    tokenizer = GreedyTokenizer.from_json("datasets/shakespeare_char/vocabulary.json")
    sampler = AutoregressiveSampler(model, tokenizer)

    prompt = "T"
    generated_text = sampler.generate_text(prompt, gpt2config.block_size - tokenizer.get_num_tokens(prompt))
    print(generated_text)


if __name__ == '__main__':
    main()
