import os
from typing import List

from sentencepiece import SentencePieceProcessor

from tokenization.tokenizer import TerminatedTokenizer, BatchTokenizer


class SentencePieceTokenizer(TerminatedTokenizer, BatchTokenizer):
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

    def encode_batch(self, texts: List[str], bos: bool = False, eos: bool = False) -> List[List[int]]:
        assert type(texts) is list

        batch_tokens = self.sp_model.encode(texts)

        if bos:
            tokens = [[self.bos_id] + t for t in batch_tokens]
        elif eos:
            tokens = [t + [self.eos_id] for t in batch_tokens]
        else:
            tokens = batch_tokens

        return tokens

    def decode_batch(self, tokens: List[List[int]], bos: bool = False, eos: bool = False) -> List[str]:
        assert type(tokens) is list
        return self.sp_model.decode(tokens)

    @property
    def vocab_size(self) -> int:
        return self.n_words

    @property
    def eot_token(self) -> int:
        return self.eos_id
