import torch
from torch.nn import Module

from tokenization.tokenizer import Tokenizer


class AutoregressiveSampler:

    def __init__(self, model: Module, tokenizer: Tokenizer):
        try:
            torch.compile(model)
        except RuntimeError as e:
            print(f"Failing to compile model: {e}")
            print("Continuing without compilation.")

        self.model = model
        self.tokenizer = tokenizer

    def generate_text(self, prompt: str, num_tokens: int, temperature: float = 1.0, top_k: int = 0,
                      include_prompt: bool = True):
        prompt_tokens = self.tokenizer.encode(prompt)
        tokens = prompt_tokens

        for _ in range(num_tokens):
            logits = self.model(torch.tensor(tokens).unsqueeze(0))
            logits = logits[0, -1, :] / temperature

            # top-k sampling
            if top_k > 0:
                top_k = min(max(top_k, 1), logits.size(-1))  # clamp top_k between 1 and vocab_size
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')

            # sample from the distribution
            token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).item()
            tokens.append(token)
        if include_prompt:
            return self.tokenizer.decode(tokens)
        else:
            return self.tokenizer.decode(tokens[len(prompt_tokens):])
