from abc import abstractmethod, ABC
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


class BatchTokenizer(Tokenizer, ABC):

    @abstractmethod
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        pass

    @abstractmethod
    def decode_batch(self, tokens: List[List[int]]) -> List[str]:
        pass


class TerminatedTokenizer(Tokenizer, ABC):

    @property
    @abstractmethod
    def eot_token(self) -> int:
        pass
