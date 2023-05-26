import numpy as np
import torch

from destillation import mass_logitify
from models.replit import ReplitLMConfig, ReplitLM
from tokenization.sentencepiece_tokenizer import SentencePieceTokenizer
from train import checkpointing


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = ReplitLMConfig(
        d_model=2560,
        n_heads=32,
        n_layers=32,
        mlp_ratio=4,
        max_seq_len=2048,
        vocab_size=32768,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        emb_pdrop=0.0,
        alibi_bias_max=8,
        use_bias=False,
        device=device,
        dtype=torch.bfloat16
    )

    model = ReplitLM(config)
    checkpointing.load_checkpoint(model, None, 'checkpoints/replit-3b', 'best', load_lazy=True)

    tokenizer = SentencePieceTokenizer("checkpoints/replit-3b/tokenizer.model")

    mass_logitify.logitify_targets(model, tokenizer, config.max_seq_len, device, "datasets/the-stack_replit/train-c-126.bin", np.uint16)


if __name__ == '__main__':
    main()
