import math
from dataclasses import dataclass
from typing import List, Callable, Optional, Dict

import torch
import torch.nn
from einops import rearrange
from models.moduleapi import BasicLanguageModel


@dataclass
class ReplitLMConfig:
    d_model: int = 2048
    n_heads: int = 16
    n_layers: int = 24
    mlp_ratio: int = 4
    max_seq_len: int = 2048
    vocab_size: int = 50368
    attn_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    emb_pdrop: float = 0.0
    alibi_bias_max: int = 8
    use_bias: bool = False
    use_kv_cache: bool = True
    device: torch.device = torch.device('cpu')
    dtype: torch.dtype = torch.float32


class LowPrecisionLayerNorm(torch.nn.LayerNorm):

    def __init__(self, normalized_shape,
                 use_bias: bool,
                 device: torch.device,
                 dtype: torch.dtype):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=1e-5,
            elementwise_affine=True,
            device=device,
            dtype=dtype,
        )
        if not use_bias:
            self.bias = None

    def forward(self, x: torch.Tensor):
        device = x.device

        downcast_x = _cast_if_autocast_enabled(x)

        # will return self, if already correct dtype.
        downcast_weight = _cast_if_autocast_enabled(self.weight)
        downcast_bias = _cast_if_autocast_enabled(self.bias)

        with torch.autocast(enabled=False, device_type=device.type):
            return torch.nn.functional.layer_norm(downcast_x, self.normalized_shape, downcast_weight, downcast_bias)


def _cast_if_autocast_enabled(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if torch.is_autocast_enabled():
        if tensor.device.type == 'cuda':
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == 'cpu':
            dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
        return tensor.to(dtype=dtype)
    return tensor


class ReplitGptMlp(torch.nn.Module):

    def __init__(self,
                 d_model: int,
                 mlp_ratio: int,
                 use_bias: bool,
                 device: torch.device,
                 dtype: torch.dtype):
        super().__init__()
        self.mlp_up = torch.nn.Linear(d_model, mlp_ratio * d_model, device=device, dtype=dtype, bias=use_bias)
        self.mlp_act = torch.nn.GELU(approximate='none')
        self.mlp_down = torch.nn.Linear(mlp_ratio * d_model, d_model, device=device, dtype=dtype, bias=use_bias)

    def forward(self, x: torch.Tensor):
        x = self.mlp_up(x)
        x = self.mlp_act(x)
        x = self.mlp_down(x)
        return x


class KvCacheEntry:

    def __init__(self, d_model: int, device: torch.device, dtype: torch.dtype):
        self.k = torch.empty((1, 0, d_model), device=device, dtype=dtype)
        self.v = torch.empty((1, 0, d_model), device=device, dtype=dtype)


class ReplitKVCache:

    def __init__(self, d_model: int, device: torch.device, dtype: torch.dtype):
        self.d_model = d_model
        self.device = device
        self.dtype = dtype
        self.kv_per_block: Dict[MultiHeadAttention, KvCacheEntry] = {}

    def __getitem__(self, item):
        if not isinstance(item, MultiHeadAttention):
            raise Exception("ReplitKVCache lookup is MultiHeadAttention instance specific")
        if item not in self.kv_per_block:
            self.kv_per_block[item] = KvCacheEntry(self.d_model, self.device, self.dtype)
        return self.kv_per_block[item]


class MultiHeadAttention(torch.nn.Module):

    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 attn_pdrop: float,
                 use_bias: bool,
                 device: torch.device,
                 dtype: torch.dtype):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.softmax_scale = 1 / math.sqrt(self.d_model / self.n_heads)
        self.attn_dropout_p = attn_pdrop

        self.Wqkv = torch.nn.Linear(self.d_model, 3 * self.d_model, device=device, dtype=dtype, bias=use_bias)

        self.out_proj = torch.nn.Linear(self.d_model, self.d_model, device=device, dtype=dtype, bias=use_bias)

    def forward(self, x: torch.Tensor,
                attn_bias: torch.Tensor,
                kv_cache: Optional[ReplitKVCache]) -> torch.Tensor:
        b, t, _ = x.shape

        qkv = self.Wqkv(x)
        q, k, v = qkv.chunk(3, dim=2)

        if kv_cache is not None:
            entry = kv_cache[self]

            k = torch.cat([entry.k, k], dim=1)
            v = torch.cat([entry.v, v], dim=1)

            entry.k = k
            entry.v = v

        y = _scaled_multihead_dot_product_attention(
            q, k, v,
            n_heads=self.n_heads,
            dropout_p=self.attn_dropout_p,
            softmax_scale=self.softmax_scale,
            is_causal=True,
            attn_bias=attn_bias
        )
        y = self.out_proj(y)
        return y


def _scaled_multihead_dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        n_heads,
        softmax_scale=None,
        attn_bias=None,
        is_causal=False,
        dropout_p=0.0,
        training=False,
        needs_weights=False,
):
    q = rearrange(query, 'b s (h d) -> b h s d', h=n_heads)
    k = rearrange(key, 'b s (h d) -> b h d s', h=n_heads)  # includes key.t()
    v = rearrange(value, 'b s (h d) -> b h s d', h=n_heads)

    min_val = torch.finfo(q.dtype).min

    b, _, s_q, d = q.shape
    s_k = k.size(-1)

    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(d)

    attn_weight = q.matmul(k) * softmax_scale

    if attn_bias is not None:
        attn_weight = attn_weight + attn_bias

    if is_causal:
        s = max(s_q, s_k)
        causal_mask = attn_weight.new_ones(s, s, dtype=torch.float16)
        causal_mask = causal_mask.tril()
        causal_mask = causal_mask.to(torch.bool)
        causal_mask = ~causal_mask
        causal_mask = causal_mask[-s_q:, -s_k:]
        attn_weight = attn_weight.masked_fill(causal_mask.view(1, 1, s_q, s_k),
                                              min_val)

    attn_weight = torch.softmax(attn_weight, dim=-1)

    if dropout_p:
        attn_weight = torch.nn.functional.dropout(attn_weight,
                                                  p=dropout_p,
                                                  training=training,
                                                  inplace=True)

    out = attn_weight.matmul(v)
    out = rearrange(out, 'b h s d -> b s (h d)')

    if needs_weights:
        return out, attn_weight
    return out


class ReplitGptBlock(torch.nn.Module):

    def __init__(self, d_model: int,
                 n_heads: int,
                 mlp_ratio: int,
                 resid_pdrop: float,
                 attn_pdrop: float,
                 use_bias: bool,
                 device: torch.device,
                 dtype: torch.dtype):
        super().__init__()
        self.ln_1 = LowPrecisionLayerNorm(
            d_model,
            use_bias=use_bias,
            device=device,
            dtype=dtype
        )
        self.attn = MultiHeadAttention(
            d_model,
            use_bias=use_bias, n_heads=n_heads,
            attn_pdrop=attn_pdrop,
            device=device,
            dtype=dtype
        )
        self.ln_2 = LowPrecisionLayerNorm(
            d_model,
            use_bias=use_bias,
            device=device,
            dtype=dtype
        )

        self.mlp = ReplitGptMlp(
            d_model=d_model,
            mlp_ratio=mlp_ratio,
            use_bias=use_bias,
            dtype=dtype,
            device=device
        )

        self.resid_attn_dropout = torch.nn.Dropout(resid_pdrop)
        self.resid_mlp_dropout = torch.nn.Dropout(resid_pdrop)

    def forward(self, x: torch.Tensor,
                attn_bias: torch.Tensor,
                kv_cache: Optional[ReplitKVCache] = None) -> torch.Tensor:
        a = self.ln_1(x)
        b = self.attn(a, attn_bias, kv_cache)
        x = x + self.resid_attn_dropout(b)
        m = self.ln_2(x)
        n = self.mlp(m)
        x = x + self.resid_mlp_dropout(n)
        return x


class AlibiPositionalEmbeddings:

    def __init__(self,
                 alibi_bias_max: int,
                 seq_length: int,
                 n_heads: int,
                 device: torch.device,
                 dtype: torch.dtype):
        alibi_bias = torch \
            .arange(1 - seq_length, 1, dtype=dtype,
                    device=device) \
            .view(1, 1, 1, seq_length)
        m = torch.arange(1, n_heads + 1, dtype=dtype, device=device)
        m = m.mul(alibi_bias_max / n_heads)

        self.alibi_bias = alibi_bias * (1. / (2 ** m.view(1, n_heads, 1, 1)))

    def make_for_seq(self, current_seq_length: int, start_pos: int = 0) -> torch.Tensor:
        return self.alibi_bias[:, :, :, -current_seq_length - start_pos:]


class ReplitLM(BasicLanguageModel):

    def __init__(self, config: ReplitLMConfig):
        super().__init__()
        self.config = config

        self.transformer = torch.nn.ModuleDict(dict(
            wte=torch.nn.Embedding(
                config.vocab_size,
                config.d_model,
                device=config.device,
                dtype=config.dtype
            ),
            emb_drop=torch.nn.Dropout(config.emb_pdrop),
            blocks=torch.nn.ModuleList([
                ReplitGptBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    mlp_ratio=config.mlp_ratio,
                    resid_pdrop=config.resid_pdrop,
                    attn_pdrop=config.attn_pdrop,
                    use_bias=config.use_bias,
                    device=config.device,
                    dtype=config.dtype,
                ) for _ in range(self.config.n_layers)
            ]),
            ln_f=LowPrecisionLayerNorm(
                config.d_model,
                use_bias=config.use_bias,
                device=config.device,
                dtype=config.dtype,
            )
        ))
        self.alibi = AlibiPositionalEmbeddings(
            config.alibi_bias_max,
            config.max_seq_len,
            config.n_heads,
            config.device,
            config.dtype
        )

    def forward(self, x: torch.Tensor, kv_cache: Optional[ReplitKVCache] = None, start_pos: int = 0) -> torch.tensor:
        b, t = x.size()
        assert t <= self.config.max_seq_len, f'Cannot forward input with seq_len={t}, this model only supports seq_len<={self.config.max_seq_len}'

        x = self.transformer.wte(x)
        x = self.transformer.emb_drop(x)

        attn_bias = self.alibi.make_for_seq(t, start_pos)

        for block in self.transformer.blocks:
            x = block(x, attn_bias, kv_cache)

        x = self.transformer.ln_f(x)

        return torch.nn.functional.linear(x, self.transformer.wte.weight, None)

    def get_probs(self, prompt: List[int], n_tokens: int, callback: Callable[[torch.Tensor], int]) -> None:
        self.eval()
        device = next(self.parameters()).device
        tokens = prompt.copy()

        kv_cache = None

        # build up kv cache
        if self.config.use_kv_cache:
            kv_cache = ReplitKVCache(
                d_model=self.config.d_model,
                device=self.config.device,
                dtype=self.config.dtype
            )
            tokens = tokens[-self.max_seq_length:]
            tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
            logits = self(tokens_tensor, kv_cache)
            logits = logits[0, -1, :]
            token = callback(logits)
            tokens.append(token)

        # forward
        for _ in range(n_tokens):
            if self.config.use_kv_cache:
                tokens_tensor = torch.tensor([tokens[-1]], dtype=torch.long, device=device).unsqueeze(0)
                logits = self(tokens_tensor, kv_cache, len(tokens) - 1)
            else:
                tokens = tokens[-self.max_seq_length:]  # crop to block size
                tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
                logits = self(tokens_tensor, kv_cache)

            logits = logits[0, -1, :]
            token = callback(logits)
            tokens.append(token)
        self.train()

    @property
    def max_seq_length(self) -> int:
        return self.config.max_seq_len

    @property
    def dtype(self) -> torch.dtype:
        return self.config.dtype
