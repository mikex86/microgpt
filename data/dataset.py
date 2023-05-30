from abc import abstractmethod
from typing import Iterator, Tuple, Callable

import numpy as np
import torch
import s3fs
from tqdm import tqdm


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


class S3FileDataset(Dataset):
    """
    TODO: This is untested. It might be broken.
    """

    def __init__(self, file_path: str, seq_len: int, token_dtype: np.dtype) -> None:
        self.file_path = file_path
        self.seq_len = seq_len
        self.token_dtype = token_dtype

    def __iter__(self):
        s3 = s3fs.S3FileSystem(anon=True)
        with s3.open(self.file_path, 'rb') as f:
            file_size = s3.du(self.file_path)
            while True:
                idx = torch.randint(low=0, high=file_size - self.seq_len - 1, size=(1,)).item()
                f.seek(idx)
                arr = np.frombuffer(f.read(self.seq_len), dtype=self.token_dtype)
                yield iter([(torch.from_numpy(arr.astype(np.int64)), \
                             torch.from_numpy(arr[1:].astype(np.int64)))])


class S3FolderDataset(Dataset):

    def __init__(self, folder_path: str, name_predicate_and_sampling_prob: Callable[[str], Tuple[bool, float]],
                 seq_len: int,
                 token_dtype: np.dtype) -> None:
        self.folder_path = folder_path
        self.name_predicate_and_sampling_prob = name_predicate_and_sampling_prob
        self.seq_len = seq_len
        self.token_dtype = token_dtype

    def __iter__(self):
        s3 = s3fs.S3FileSystem(anon=True)

        files = []
        probs = []
        for file_name in tqdm(s3.ls(self.folder_path), desc="Listing s3 files..."):
            should_read, sampling_probability = self.name_predicate_and_sampling_prob(file_name)
            if should_read:
                f = s3.open(file_name, 'rb')
                file_size = s3.du(file_name)
                files.append((f, file_size))
                probs.append(sampling_probability)

        # Normalize probabilities
        probs = torch.tensor(probs)
        probs /= probs.sum()

        dtype_size = np.dtype(self.token_dtype).itemsize

        while True:
            idx = torch.multinomial(probs, 1).item()
            ftup = files[idx]
            f, file_size = ftup
            if file_size < (self.seq_len + 1) * dtype_size:
                continue
            idx = torch.randint(low=0, high=file_size - (self.seq_len + 1) * dtype_size - 1, size=(1,)).item()
            if idx % dtype_size != 0:
                idx -= idx % dtype_size
            if idx < 0:
                idx = 0
            f.seek(idx)
            arr = np.frombuffer(f.read((self.seq_len + 1) * dtype_size), dtype=self.token_dtype)
            yield iter([(torch.from_numpy(arr[:-1].astype(np.int64)), \
                         torch.from_numpy(arr[1:].astype(np.int64)))])
