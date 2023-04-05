from typing import List, Callable

import torch.nn
from dataclasses import dataclass

from torch import nn
from torch.cuda.amp import GradScaler

from models.moduleapi import ISparselyWeightDecayedModule, WeightDecayGroups, ILanguageModel
from tokenization.tokenizer import Tokenizer
import tiktoken


@dataclass
class Gpt2Config:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = torch.float32
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to the nearest multiple of 64 for efficiency
    n_layers: int = 12
    n_heads: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linear layers and LayerNorms, like GPT-2. False: a bit better and faster


class Gpt2LayerNorm(torch.nn.Module):

    def __init__(self, ndim: int, bias: bool, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(ndim, device=device, dtype=dtype))
        self.bias = torch.nn.Parameter(torch.zeros(ndim, device=device, dtype=dtype)) if bias else None

    def forward(self, x):
        return torch.nn.functional.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)


class Gpt2CausalSelfAttention(torch.nn.Module):

    def __init__(self, config: Gpt2Config):
        super().__init__()
        assert config.n_embd % config.n_heads == 0

        # key, query, value projections for all heads
        self.c_attn = torch.nn.Linear(config.n_embd, config.n_embd * 3, bias=config.bias, device=config.device, dtype=config.dtype)

        # output projection
        self.c_proj = torch.nn.Linear(config.n_embd, config.n_embd, bias=config.bias, device=config.device, dtype=config.dtype)

        # regularization
        self.attn_drop = torch.nn.Dropout(config.dropout)
        self.resid_drop = torch.nn.Dropout(config.dropout)

        self.n_head = config.n_heads
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        b, t, c = x.size()

        # calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)  # q, k, v all are of shape (b, t, n_embd)

        # split hidden into heads
        q = q.view(b, t, self.n_head, c // self.n_head)
        k = k.view(b, t, self.n_head, c // self.n_head)
        v = v.view(b, t, self.n_head, c // self.n_head)

        # transpose to get (b, n_head, t, c // n_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = torch.nn.functional.scaled_dot_product_attention(q, k, v,
                                                             attn_mask=None,
                                                             dropout_p=self.dropout,
                                                             is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, c)  # reassemble across non-contiguous dimensions (b, t, c)

        # output projection
        y = self.resid_drop(self.c_proj(y))
        return y


class Gpt2MLP(torch.nn.Module):

    def __init__(self, config: Gpt2Config):
        super().__init__()
        self.c_fc = torch.nn.Linear(config.n_embd, config.n_embd * 4, bias=config.bias, device=config.device, dtype=config.dtype)
        self.c_proj = torch.nn.Linear(config.n_embd * 4, config.n_embd, bias=config.bias, device=config.device, dtype=config.dtype)
        self.act = torch.nn.GELU()
        self.drop = torch.nn.Dropout(config.dropout)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.drop(h2)


class Gpt2Block(torch.nn.Module):

    def __init__(self, config: Gpt2Config):
        super().__init__()

        self.ln_1 = Gpt2LayerNorm(config.n_embd, bias=config.bias, device=config.device, dtype=config.dtype)
        self.attn = Gpt2CausalSelfAttention(config)
        self.ln_2 = Gpt2LayerNorm(config.n_embd, bias=config.bias, device=config.device, dtype=config.dtype)
        self.mlp = Gpt2MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Gpt2Model(ISparselyWeightDecayedModule, ILanguageModel):

    def __init__(self, config: Gpt2Config):
        super().__init__()

        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd, dtype=config.dtype, device=config.device)
        self.wpe = nn.Embedding(config.block_size, config.n_embd, dtype=config.dtype, device=config.device)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Gpt2Block(config) for _ in range(config.n_layers)])
        self.ln_f = Gpt2LayerNorm(config.n_embd, bias=config.bias, device=config.device, dtype=config.dtype)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias, device=config.device, dtype=config.dtype)

    def forward(self, x):
        b, t = x.size()
        assert t <= self.config.block_size, "Cannot forward, model block size is exhausted."

        # token embeddings
        tok_emb = self.wte(x)

        # add position embeddings
        pos_idx = torch.arange(0, t, device=x.device).unsqueeze(0)
        pos_emb = self.wpe(pos_idx)
        x = self.drop(tok_emb + pos_emb)

        # transformer
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.lm_head(x)

    def get_probs(self, prompt: List[int], n_tokens: int, callback: Callable[[torch.tensor], int]) -> None:
        self.eval()
        device = next(self.parameters()).device
        tokens = prompt.copy()
        with torch.no_grad():
            for _ in range(n_tokens):
                tokens = tokens[-self.config.block_size:]  # crop to block size
                tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
                logits = self(tokens_tensor)
                logits = logits[0, -1, :]
                token = callback(logits)
                tokens.append(token)
        self.train()

    def get_weight_decay_groups(self) -> WeightDecayGroups:
        all_modules = list(self.named_modules())
        no_weight_decay_params = []

        # weight decay on all parameters except bias and LayerNorms
        for name, module in all_modules:
            if isinstance(module, torch.nn.Linear):
                no_weight_decay_params.append(module.bias)
            elif isinstance(module, Gpt2LayerNorm):
                no_weight_decay_params.extend([module.weight, module.bias])

        weight_decay_params = []

        for param in self.parameters():
            # if there is no param in no_weight_decay_params wrapping the same tensor as param, we add it to the list
            if not any([param is p for p in no_weight_decay_params]):
                weight_decay_params.append(param)

        return WeightDecayGroups(
            weight_decay_params=weight_decay_params,
            no_weight_decay_params=no_weight_decay_params
        )

    def back_propagate(self, x: torch.tensor, targets: torch.tensor, loss_scalar: GradScaler = None,
                       hyper_save_memory: bool = False) -> float:
        self.train()
        device = next(self.parameters()).device
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        unscaled_loss = loss.item()

        if loss_scalar is not None:
            loss = loss_scalar.scale(loss)
        loss.backward()

        if hyper_save_memory:
            del logits
            del loss
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        return unscaled_loss

    @torch.no_grad()
    def get_eval_loss(self, x: torch.tensor, y: torch.tensor) -> float:
        self.eval()
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss_item = loss.item()

        del logits
        del loss

        return loss_item

    @property
    def dtype(self) -> torch.dtype:
        return self.config.dtype


class Gpt2Tokenizer(Tokenizer):

    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.n_vocab
