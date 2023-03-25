import json
from typing import List, Dict

from tokenization.tokenizer import Tokenizer


class GreedyTokenizer(Tokenizer):
    """
    A tokenizer that greedily consumes the most course grained token from the text.
    This is a very simple and slow tokenizer. You should never use it to encode large texts!
    """

    def __init__(self, stoi: Dict[str, int]):
        self.stoi = stoi
        self.itos = {i: token for token, i in stoi.items()}

    @staticmethod
    def from_json(json_file_path: str):
        with open(json_file_path, "r") as f:
            possible_tokens = json.load(f)
        stoi = {token: i for i, token in enumerate(possible_tokens)}
        return GreedyTokenizer(stoi)

    def encode(self, text: str) -> List[int]:
        # iteratively consume the most course grained token from the text
        # and replace it with the corresponding token id
        tokens = []
        while len(text) > 0:
            for i in range(len(text), 0, -1):
                token = text[:i]
                if token in self.stoi:
                    tokens.append(self.stoi[token])
                    text = text[i:]
                    break
            else:
                raise ValueError(f"Unable to tokenize {text}")
        return tokens

    def decode(self, tokens: List[int]) -> str:
        return "".join(self.itos[token] for token in tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)
