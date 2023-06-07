from typing import List, Callable, Union

import torch

from models.moduleapi import ILanguageModel
from tokenization.tokenizer import Tokenizer


class AutoregressiveSampler:

    def __init__(self, model: ILanguageModel, tokenizer: Tokenizer, token_blacklist: List[int] = None):
        try:
            torch.compile(model)
        except RuntimeError as e:
            print(f"Failing to compile model: {e}")
            print("Continuing without compilation.")

        self.model = model
        self.tokenizer = tokenizer
        self.token_blacklist = token_blacklist

    @torch.inference_mode()
    def __sample(self, prompt_tokens: List[int], num_tokens: int, temperature: float, top_k: int,
                 token_callback: Callable[[int], bool] = None):
        new_tokens = []
        tokens = prompt_tokens.copy()

        def handle_props(logits: torch.tensor):
            nonlocal top_k, token_callback

            # sample token
            if temperature == 0.0:
                logits = torch.softmax(logits, dim=-1)
                token = torch.argmax(logits, dim=-1).item()
            else:
                logits /= temperature

                # top-k sampling
                if top_k > 0:
                    top_k = min(max(top_k, 1), logits.size(-1))  # clamp top_k between 1 and vocab_size
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = -float('Inf')

                # blacklisting
                if self.token_blacklist is not None:
                    for token in self.token_blacklist:
                        logits[token] = -float('Inf')

                # sample from the distribution
                token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).item()

            # add token to tokens
            tokens.append(token)
            new_tokens.append(token)

            # invoke token callback
            if token_callback is not None:
                if not token_callback(token):
                    return token, True  # should_stop = True

            # return token such that the model can continue
            return token, False  # should_stop = False

        self.model.get_probs(
            prompt_tokens, num_tokens,
            lambda probs: handle_props(probs)
        )

        return new_tokens

    def generate_text(self, prompt: str, num_tokens: int, temperature: float = 1.0, top_k: int = 0,
                      include_prompt: bool = True,
                      stop: Union[str, List[str]] = None) -> str:
        if stop is None:
            stop = []

        stop_sequences = [self.tokenizer.encode(stop_sequence) for stop_sequence in stop]

        prompt_tokens = self.tokenizer.encode(prompt)

        tokens = prompt_tokens.copy()

        def check_stop(token: int) -> bool:
            tokens.append(token)

            for stop_sequence in stop_sequences:
                if tokens[-len(stop_sequence):] == stop_sequence:
                    return False
            return True

        new_tokens = self.__sample(prompt_tokens, num_tokens, temperature, top_k, lambda token: check_stop(token))

        # remove trailing stop sequences
        for stop_sequence in stop_sequences:
            if new_tokens[-len(stop_sequence):] == stop_sequence:
                new_tokens = new_tokens[:-len(stop_sequence)]

        return self.tokenizer.decode(prompt_tokens + new_tokens) if include_prompt else self.tokenizer.decode(
            new_tokens)

    def stream_text(self, prompt: str, num_tokens: int,
                    text_callback: Callable[[str], None],
                    temperature: float = 1.0,
                    top_k: int = 0,
                    stop: Union[str, List[str]] = None):
        if stop is None:
            stop = []

        stop_sequences = [self.tokenizer.encode(stop_sequence) for stop_sequence in stop]

        max_stop_sequence_len = \
            max([len(stop_sequence) for stop_sequence in stop_sequences]) if len(stop_sequences) > 0 else 0

        prompt_tokens = self.tokenizer.encode(prompt)
        tokens = prompt_tokens.copy()

        prev_str = prompt

        def handle_token(token: int):
            nonlocal prev_str, tokens
            tokens.append(token)

            # only publish tokens that are definitely not part of a stop sequence
            # that's why we only publish tokens that are at least max_stop_sequence_len tokens away from the current
            # end.
            safe_tokens = tokens[:-max_stop_sequence_len] if max_stop_sequence_len > 0 else tokens
            new_str = self.tokenizer.decode(safe_tokens)
            str_to_publish = new_str[len(prev_str):]
            if len(str_to_publish) > 0:
                text_callback(str_to_publish)
                prev_str = new_str

                for stop_sequence in stop_sequences:
                    if tokens[-len(stop_sequence):] == stop_sequence:
                        # flush remaining tokens
                        safe_tokens = safe_tokens[:-len(stop_sequence)]
                        new_str = self.tokenizer.decode(safe_tokens)
                        str_to_publish = new_str[len(prev_str):]
                        if len(str_to_publish) > 0:
                            text_callback(str_to_publish)
                        prev_str = new_str
                        return False

            return True

        self.__sample(
            prompt_tokens, num_tokens, temperature, top_k,
            lambda token: handle_token(token)
        )
