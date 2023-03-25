from abc import abstractmethod
from typing import List


class Tokenizer:

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass

    def get_num_tokens(self, text: str):
        return len(self.encode(text))

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        pass
