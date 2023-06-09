# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by mikex86
# This software may be used and distributed according to the terms of the GNU General Public License version 3.
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Callable

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

from utils import torchhacks
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)
from sentencepiece import SentencePieceProcessor
from torch import nn

from models.moduleapi import ILanguageModel
from tokenization.tokenizer import Tokenizer
from utils import torchhooks

# needed for fp16 on cpu (temporary upcast to fp32)
torchhooks.init_torch_hooks()


@dataclass
class LlamaConfig:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048
    init_weights: bool = False  # wether to initialize weights with normal distribution

    @staticmethod
    def from_json(file_path: str) -> 'LlamaConfig':
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return LlamaConfig(**config_dict)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: LlamaConfig, target_device: torch.device):
        super().__init__()

        self.n_local_heads = args.n_heads // fs_init.get_model_parallel_world_size()
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=(lambda x: torch.nn.init.normal_(x, std=0.02)) if args.init_weights else (lambda x: x),
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=(lambda x: torch.nn.init.normal_(x, std=0.02)) if args.init_weights else (lambda x: x),
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=(lambda x: torch.nn.init.normal_(x, std=0.02)) if args.init_weights else (lambda x: x),
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=(lambda x: torch.nn.init.normal_(x, std=0.02)) if args.init_weights else (lambda x: x),
        )

        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim),
            device=target_device,
        )
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim),
            device=target_device,
        )

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # check if is eval
        if not self.training:
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos: start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos: start_pos + seqlen] = xv

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else:
            keys = xk
            values = xv

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
            self,
            args: LlamaConfig,
            dim: int,
            hidden_dim: int,
            multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False,
            init_method=(lambda x: torch.nn.init.normal_(x, std=0.02)) if args.init_weights else (lambda x: x),
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True,
            init_method=(lambda x: torch.nn.init.normal_(x, std=0.02)) if args.init_weights else (lambda x: x),
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False,
            init_method=(lambda x: torch.nn.init.normal_(x, std=0.02)) if args.init_weights else (lambda x: x),
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):

    def __init__(self, layer_id: int, args: LlamaConfig, target_device: torch.device):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args, target_device)
        self.feed_forward = FeedForward(
            args=args, dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class LlamaModel(ILanguageModel):

    def __init__(self, args: LlamaConfig, target_device: torch.device):
        super().__init__()
        self.params = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        self.tok_embeddings = ParallelEmbedding(
            args.vocab_size, args.dim,
            init_method=(lambda x: torch.nn.init.normal_(x)) if args.init_weights else (lambda x: x),
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(layer_id, args, target_device))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = ColumnParallelLinear(
            args.dim, args.vocab_size, bias=False,
            init_method=(lambda x: torch.nn.init.normal_(x, std=0.02)) if args.init_weights else (lambda x: x),
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    @staticmethod
    def load(checkpoint_path: str,
             tokenizer: 'SentencePieceTokenizer',
             target_device: torch.device,
             local_rank: int,
             world_size: int,
             fp_16: bool = True) -> 'LlamaModel':
        checkpoints = sorted(Path(checkpoint_path).glob("*.pth"))

        if len(checkpoints) != world_size:
            raise ValueError(
                f"Found {len(checkpoints)} checkpoints but expected {world_size}"
            )

        checkpoint = checkpoints[local_rank]

        config = LlamaConfig.from_json(os.path.join(checkpoint_path, "params.json"))

        # modify vocab size to match tokenizer
        config.vocab_size = tokenizer.vocab_size

        if fp_16:
            is_cuda = target_device.type == "cuda"

            if is_cuda:
                torch.set_default_tensor_type(torch.cuda.HalfTensor)
            else:
                torch.set_default_tensor_type(torch.HalfTensor)

        model = LlamaModel(config, target_device).to(target_device)

        # hack to lazily load weights and avoid OOM
        state_dict = torchhacks.lazy_load(checkpoint)
        # state_dict = torch.load(checkpoint, map_location=target_device)

        model.load_state_dict(state_dict, strict=False)

        if fp_16:
            torch.set_default_tensor_type(torch.FloatTensor)

        del state_dict

        return model

    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = h.type_as(self.output.weight)

        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h)
        return output.float()

    @torch.inference_mode()
    def get_probs(self, prompt: List[int], n_tokens: int, callback: Callable[[torch.tensor], Tuple[int, bool]]) -> None:
        self.eval()
        device = next(self.parameters()).device
        tokens = prompt.copy()

        prev_pos = 0
        cur_pos = len(tokens)
        for n in range(n_tokens):
            tokens_tensor = torch.tensor(tokens[prev_pos:cur_pos], dtype=torch.long, device=device).unsqueeze(0)
            logits = self(tokens_tensor, start_pos=prev_pos)[:, -1, :]
            token, should_stop = callback(logits)
            tokens.append(token)
            prev_pos = cur_pos
            cur_pos += 1
            if should_stop:
                break
        self.train()

    def back_propagate_targets(self, x: torch.tensor, targets: torch.tensor, loss_scalar: GradScaler = None,
                               hyper_save_memory: bool = False) -> Tuple[float, torch.Tensor]:
        self.train()
        device = next(self.parameters()).device
        logits = self(x, start_pos=0)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        unscaled_loss = loss.item()

        if loss_scalar is not None:
            loss = loss_scalar.scale(loss)
        loss.backward()

        logits_copy = logits.detach().clone()

        if hyper_save_memory:
            del logits
            del loss
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        return unscaled_loss, logits_copy

    @torch.no_grad()
    def get_eval_loss(self, x: torch.tensor, y: torch.tensor) -> float:
        self.eval()
        logits = self(x, start_pos=0)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss_item = loss.item()

        del logits
        del loss

        return loss_item

    @property
    def dtype(self) -> torch.dtype:
        return self.output.weight.dtype

