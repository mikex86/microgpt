from typing import Tuple

import torch

from models.moduleapi import ILanguageModel
from models.replit import ReplitLMConfig, ReplitLM
from rest import lm_rest
from tokenization.sentencepiece_tokenizer import SentencePieceTokenizer
from tokenization.tokenizer import Tokenizer
from train import checkpointing

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _load_replit_3b() -> Tuple[ILanguageModel, Tokenizer]:
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
    return model, tokenizer


lm_rest.models = {
    "replit-3b": lm_rest.ServedModel(*_load_replit_3b())
}

if __name__ == '__main__':
    lm_rest.app.run()
