from abc import abstractmethod
from typing import Iterator, Tuple

import numpy as np
import torch
import s3fs


class Dataset:

    @abstractmethod
    def __iter__(self) -> Iterator[Iterator[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        :return: an iterator over iterators over (input, target) pairs.
        It returns an iterator over "sequences" of (input, target) pairs,
        where each sequence's set of (input, target) pairs must be processed in order as to not distort the dataset.
        """
        pass


class BinaryTokenDataset(Dataset):

    def __init__(self, file_path: str, seq_len: int, token_dtype: np.dtype) -> None:
        self.array = np.memmap(file_path, mode="r", dtype=token_dtype)
        self.seq_len = seq_len

    def __iter__(self):
        while True:
            idx = torch.randint(low=0, high=len(self.array) - self.seq_len - 1, size=(1,)).item()
            yield iter([(torch.from_numpy(self.array[idx:idx + self.seq_len].astype(np.int64)), \
                         torch.from_numpy(self.array[idx + 1:idx + self.seq_len + 1].astype(np.int64)))])


class S3Dataset(Dataset):
    """
    TODO: This is untested. It might be broken.
    """

    def __init__(self, file_path: str, seq_len: int, token_dtype: np.dtype) -> None:
        self.file_path = file_path
        self.seq_len = seq_len
        self.token_dtype = token_dtype

    def __iter__(self):
        s3 = s3fs.S3FileSystem()
        with s3.open(self.file_path, 'rb') as f:
            file_size = s3.du(self.file_path)
            while True:
                idx = torch.randint(low=0, high=file_size - self.seq_len - 1, size=(1,)).item()
                f.seek(idx)
                arr = np.frombuffer(f.read(self.seq_len), dtype=self.token_dtype)
                yield iter([(torch.from_numpy(arr.astype(np.int64)), \
                             torch.from_numpy(arr[1:].astype(np.int64)))])
