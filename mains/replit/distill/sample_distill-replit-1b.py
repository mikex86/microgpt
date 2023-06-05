import torch

from inference.sampler import AutoregressiveSampler
from models.replit import ReplitLMConfig, ReplitLM
from tokenization.sentencepiece_tokenizer import SentencePieceTokenizer
from train import checkpointing


def main():
    # device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = ReplitLMConfig(
        d_model=1536,
        n_heads=12,
        n_layers=12,
        mlp_ratio=4,
        max_seq_len=2048,
        vocab_size=32768,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        emb_pdrop=0.0,
        alibi_bias_max=8,
        use_bias=False,
        device=device,
        dtype=torch.float32
    )

    model = ReplitLM(config)
    checkpointing.load_checkpoint(model, None, 'checkpoints/replit-distill-1b', 'best', load_lazy=True)

    tokenizer = SentencePieceTokenizer("checkpoints/replit-distill-1b/tokenizer.model")
    sampler = AutoregressiveSampler(model, tokenizer)

    prompt = "class Gpt2(torch.nn.Module):\n"

    sampler.stream_text(
        prompt,
        1000,
        lambda token_str: print(token_str, end=''),
        temperature=0.0
    )


if __name__ == '__main__':
    main()
