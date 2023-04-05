import math
from typing import List, Callable

import torch.nn
from dataclasses import dataclass

from torch import nn
from torch.cuda.amp import GradScaler

from models.moduleapi import ISparselyWeightDecayedModule, WeightDecayGroups, ILanguageModel


@dataclass
class MemGptConfig:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    block_size: int = 1024
    n_windows: int = 4
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to the nearest multiple of 64 for efficiency
    n_layers: int = 12
    n_heads: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linear layers and LayerNorms, like GPT-2. False: a bit better and faster


class MemGptLayerNorm(torch.nn.Module):

    def __init__(self, ndim: int, bias: bool, device: torch.device):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(ndim, device=device))
        self.bias = torch.nn.Parameter(torch.zeros(ndim, device=device)) if bias else None

    def forward(self, x):
        return torch.nn.functional.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)


class MemGptTemporalCrossAttention(torch.nn.Module):

    def __init__(self, config: MemGptConfig):
        super().__init__()
        self.config = config

        assert config.n_embd % config.n_heads == 0

        # query, value projections for all heads
        self.now_key_enc = torch.nn.Linear(config.n_embd, config.n_embd, bias=config.bias, device=config.device)

        # key, value projection for prev x
        self.c_attn_prev = torch.nn.Linear(config.n_embd, config.n_embd * 2, bias=config.bias, device=config.device)

        # output projection
        self.c_proj = torch.nn.Linear(config.n_embd, config.n_embd, bias=config.bias, device=config.device)

        # regularization
        self.attn_drop = torch.nn.Dropout(config.dropout)
        self.resid_drop = torch.nn.Dropout(config.dropout)

        self.n_head = config.n_heads
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.flash = True

        if not self.flash:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.bias = torch.tril(torch.ones(config.block_size, config.block_size)) \
                .view(1, 1, config.block_size, config.block_size)

    def forward(self, x, prev_x):
        b, t, c = x.size()

        # calculate query, key, values for all heads in batch
        q = self.now_key_enc(x)  # (b, t, n_embd)
        k, v = self.c_attn_prev(prev_x).split(self.n_embd, dim=2)  # (b, block_size, n_embd)

        # split hidden into heads
        q = q.view(b, t, self.n_head, c // self.n_head)
        k = k.view(b, self.config.block_size, self.n_head, c // self.n_head)
        v = v.view(b, self.config.block_size, self.n_head, c // self.n_head)

        # transpose to get (b, n_head, t, c // n_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # self attention
        if self.flash and self.training:
            # only for training!
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v,
                                                                 attn_mask=None,
                                                                 dropout_p=self.dropout,
                                                                 is_causal=False)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = torch.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v  # (B, nh, t, block_size) x (B, nh, block_size, hs) -> (B, nh, block_size, hs)

        # reassemble across non-contiguous dimensions (b, block_size, c)
        y = y.transpose(1, 2).contiguous().view(b, t, c)

        # output projection
        y = self.resid_drop(self.c_proj(y))
        return y


class MemGptCausalSelfAttention(torch.nn.Module):

    def __init__(self, config: MemGptConfig):
        super().__init__()
        assert config.n_embd % config.n_heads == 0

        # key, query, value projections for all heads
        self.c_attn = torch.nn.Linear(config.n_embd, config.n_embd * 3, bias=config.bias, device=config.device)

        # output projection
        self.c_proj = torch.nn.Linear(config.n_embd, config.n_embd, bias=config.bias, device=config.device)

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


class MemGptMLP(torch.nn.Module):

    def __init__(self, config: MemGptConfig):
        super().__init__()
        self.c_fc = torch.nn.Linear(config.n_embd, config.n_embd * 4, bias=config.bias, device=config.device)
        self.c_proj = torch.nn.Linear(config.n_embd * 4, config.n_embd, bias=config.bias, device=config.device)
        self.act = torch.nn.GELU()
        self.drop = torch.nn.Dropout(config.dropout)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.drop(h2)


class MemGptCrossTemporalBlock(torch.nn.Module):

    def __init__(self, config: MemGptConfig):
        super().__init__()

        self.ln_11 = MemGptLayerNorm(config.n_embd, bias=config.bias, device=config.device)
        self.ln_12 = MemGptLayerNorm(config.n_embd, bias=config.bias, device=config.device)
        self.attn_1 = MemGptTemporalCrossAttention(config)

        self.ln_21 = MemGptLayerNorm(config.n_embd, bias=config.bias, device=config.device)
        self.ln_22 = MemGptLayerNorm(config.n_embd, bias=config.bias, device=config.device)
        self.mlp = MemGptMLP(config)

    def forward(self, x, prev_x):
        x = x + self.attn_1(self.ln_11(x), self.ln_12(prev_x))  # (b, t, c)
        x = x + self.mlp(self.ln_21(x))  # (b, t, c)
        return x


class MemGptDefaultBlock(torch.nn.Module):

    def __init__(self, config: MemGptConfig):
        super().__init__()

        self.ln_1 = MemGptLayerNorm(config.n_embd, bias=config.bias, device=config.device)
        self.attn = MemGptCausalSelfAttention(config)
        self.ln_2 = MemGptLayerNorm(config.n_embd, bias=config.bias, device=config.device)
        self.mlp = MemGptMLP(config)

    def forward(self, x, _):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MemGptModel(ISparselyWeightDecayedModule, ILanguageModel):

    def __init__(self, config: MemGptConfig):
        super().__init__()

        self.config = config

        self.init_prev_layer_acts = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(1, config.block_size, config.n_embd, device=config.device))
            for _ in range(config.n_layers)
        ])

        self.wte = nn.Embedding(config.vocab_size, config.n_embd, device=config.device)
        self.wpe = nn.Embedding(config.block_size, config.n_embd, device=config.device)
        self.bwpe = nn.Embedding(config.n_windows, config.n_embd, device=config.device)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([
            MemGptDefaultBlock(config) if i % 2 == 0 else MemGptCrossTemporalBlock(config)
            for i in range(config.n_layers)
        ])
        self.ln_f = MemGptLayerNorm(config.n_embd, bias=config.bias, device=config.device)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias, device=config.device)

    def forward(self, x):
        b, t = x.size()

        # split into chunks of block size
        chunks = []
        n_chunks = t // self.config.block_size + int(t % self.config.block_size != 0)
        for i in range(n_chunks):
            chunks.append(x[:, i * self.config.block_size:(i + 1) * self.config.block_size])

        prev_layer_acts = [
            self.init_prev_layer_acts[i]
            .expand(b, self.config.block_size, self.config.n_embd)
            for i in range(self.config.n_layers)
        ]
        output = []

        max_t1 = len(chunks)

        assert max_t1 <= self.config.n_windows, f"t1={max_t1} > num_blocks={self.config.n_windows}"

        for t1 in range(max_t1):
            x_chunk = chunks[t1]  # (b, t2)
            t2 = x_chunk.size(1)

            assert t2 <= self.config.block_size, f"t2={t2} > block_size={self.config.block_size}"

            # token embeddings
            tok_emb = self.wte(x_chunk)

            # add position embeddings
            pos_idx = torch.arange(0, t2, device=x_chunk.device).unsqueeze(0)
            pos_emb = self.wpe(pos_idx) + self.bwpe(torch.tensor(t1, device=x.device))
            h = self.drop(tok_emb + pos_emb)

            # transformer
            for i in range(self.config.n_layers):
                block = self.blocks[i]
                prev_h = prev_layer_acts[i]
                h = block(h, prev_h)
                prev_layer_acts[i] = h

            h = self.ln_f(h)  # (b, t2, c)
            output.append(h)

        h = torch.cat(output, dim=1)  # (b, t1 * t2, c)
        return self.lm_head(h)

    def back_propagate(self, x: torch.tensor, targets: torch.tensor,
                       loss_scalar: GradScaler = None,
                       hyper_save_memory: bool = False) -> float:
        self.train()
        device = next(self.parameters()).device
        b, t = x.size()

        # split x into chunks of block size
        x_chunks = []
        n_chunks = t // self.config.block_size + int(t % self.config.block_size != 0)
        for i in range(n_chunks):
            x_chunks.append(x[:, i * self.config.block_size:(i + 1) * self.config.block_size])

        # split targets into chunks of block size
        targets_chunks = []
        for i in range(n_chunks):
            targets_chunks.append(targets[:, i * self.config.block_size:(i + 1) * self.config.block_size])

        prev_layer_acts = [
            self.init_prev_layer_acts[i]
            .expand(b, self.config.block_size, self.config.n_embd)
            for i in range(self.config.n_layers)
        ]
        max_t1 = len(x_chunks)

        assert max_t1 <= self.config.n_windows, f"t1={max_t1} > num_blocks={self.config.n_windows}"

        losses = []

        for t1 in range(max_t1):
            x_chunk = x_chunks[t1]  # (b, t2)
            t2 = x_chunk.size(1)

            assert t2 <= self.config.block_size, f"t2={t2} > block_size={self.config.block_size}"

            # token embeddings
            tok_emb = self.wte(x_chunk)

            # add position embeddings
            pos_idx = torch.arange(0, t2, device=x_chunk.device).unsqueeze(0)
            pos_emb = self.wpe(pos_idx) + self.bwpe(torch.tensor(t1, device=x.device))
            h = self.drop(tok_emb + pos_emb)

            # transformer
            for i in range(self.config.n_layers):
                block = self.blocks[i]
                prev_h = prev_layer_acts[i]
                h = block(h, prev_h)
                prev_layer_acts[i] = h

            h = self.ln_f(h)  # (b, t2, c)

            # compute loss
            lm_logits = self.lm_head(h)  # (b, t2, vocab_size)
            targets_chunk = targets_chunks[t1]
            loss = torch.nn.functional.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), targets_chunk.reshape(-1))

            # append loss number to list for reporting
            losses.append(loss.item())

            # scale loss if using mixed precision
            if loss_scalar is not None:
                loss = loss_scalar.scale(loss)

            loss.backward(retain_graph=True)

            # free memory
            if hyper_save_memory:
                del lm_logits
                del loss
                # free cuda cache
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        # return mean un-scaled loss
        return sum(losses) / len(losses)

    def get_probs(self, prompt: List[int], n_tokens: int, callback: Callable[[torch.tensor], int]) -> None:
        self.eval()
        device = next(self.parameters()).device
        prompt = prompt.copy()
        b = 1  # batch size
        with torch.no_grad():
            prev_layer_acts = [
                self.init_prev_layer_acts[i]
                .expand(b, self.config.block_size, self.config.n_embd)
                for i in range(self.config.n_layers)
            ]
            for _ in range(n_tokens):
                prompt_tensor = torch.tensor(prompt, device=device).view(1, -1)
                t = prompt_tensor.size(1)  # total sequence length

                # split into chunks of block size
                chunks = []
                n_chunks = t // self.config.block_size + int(t % self.config.block_size != 0)
                for i in range(n_chunks):
                    chunks.append(prompt_tensor[:, i * self.config.block_size:(i + 1) * self.config.block_size])

                t1 = len(chunks)

                assert t1 <= self.config.n_windows, f"t1={t1} > num_blocks={self.config.n_windows}"

                # only compute model in last chunk!
                x_chunk = chunks[t1 - 1]
                t2 = x_chunk.size(1)

                assert t2 <= self.config.block_size, f"t2={t2} > block_size={self.config.block_size}"

                # token embeddings
                tok_emb = self.wte(x_chunk)

                # add position embeddings
                pos_idx = torch.arange(0, t2, device=x_chunk.device).unsqueeze(0)
                pos_emb = self.wpe(pos_idx) + self.bwpe(torch.tensor(t1 - 1, device=device))
                h = self.drop(tok_emb + pos_emb)

                # transformer
                for layer_idx in range(self.config.n_layers):
                    block = self.blocks[layer_idx]
                    prev_h = prev_layer_acts[layer_idx]
                    h = block(h, prev_h)

                    # only switch prev_h if the chunk is full
                    if t2 == self.config.block_size:
                        prev_layer_acts[layer_idx] = h

                h = self.ln_f(h)  # (b, t2, c)

                # get probs
                probs = self.lm_head(h)
                probs = probs[0, -1, :]  # (b, c)
                chosen_token = callback(probs)

                # add chosen token to prompt
                prompt.append(chosen_token)
        self.train()

    def get_weight_decay_groups(self) -> WeightDecayGroups:
        all_modules = list(self.named_modules())
        no_weight_decay_params = []

        # weight decay on all parameters except bias and LayerNorms
        for name, module in all_modules:
            if isinstance(module, torch.nn.Linear):
                no_weight_decay_params.append(module.bias)
            elif isinstance(module, MemGptLayerNorm):
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
