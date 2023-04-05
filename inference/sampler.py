from typing import List, Callable

import torch

from models.moduleapi import ILanguageModel
from tokenization.tokenizer import Tokenizer


class AutoregressiveSampler:

    def __init__(self, model: ILanguageModel, tokenizer: Tokenizer):
        try:
            torch.compile(model)
        except RuntimeError as e:
            print(f"Failing to compile model: {e}")
            print("Continuing without compilation.")

        self.model = model
        self.tokenizer = tokenizer

    def __sample(self, prompt_tokens: List[int], num_tokens: int, temperature: float, top_k: int,
                 token_callback: Callable[[int], None] = None):
        new_tokens = []
        tokens = prompt_tokens.copy()

        def handle_props(logits: torch.tensor):
            nonlocal top_k, token_callback
            logits /= temperature

            # top-k sampling
            if top_k > 0:
                top_k = min(max(top_k, 1), logits.size(-1))  # clamp top_k between 1 and vocab_size
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')

            # sample from the distribution
            token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).item()
            tokens.append(token)
            new_tokens.append(token)

            # invoke token callback
            if token_callback is not None:
                token_callback(token)

            # return token such that the model can continue
            return token

        self.model.get_probs(
            prompt_tokens, num_tokens,
            lambda probs: handle_props(probs)
        )

        return new_tokens

    def generate_text(self, prompt: str, num_tokens: int, temperature: float = 1.0, top_k: int = 0,
                      include_prompt: bool = True):
        prompt_tokens = self.tokenizer.encode(prompt)
        new_tokens = self.__sample(prompt_tokens, num_tokens, temperature, top_k)
        return self.tokenizer.decode(prompt_tokens + new_tokens) if include_prompt else self.tokenizer.decode(
            new_tokens)

    def stream_text(self, prompt: str, num_tokens: int,
                    token_callback: Callable[[str], None],
                    temperature: float = 1.0,
                    top_k: int = 0):
        prompt_tokens = self.tokenizer.encode(prompt)
        self.__sample(
            prompt_tokens, num_tokens, temperature, top_k,
            lambda token: token_callback(self.tokenizer.decode([token]))
        )
