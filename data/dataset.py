import multiprocessing
import os
import queue
import time
from abc import abstractmethod
from collections import Counter
from typing import Iterator, Tuple, Callable, Optional, List

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


class S3AsyncReader(multiprocessing.Process):

    def __init__(self, file_names: List[str], file_sizes: List[int], token_dtype: np.dtype, block_size: int,
                 rx_queue: multiprocessing.Queue,
                 tx_queue: multiprocessing.Queue):
        super().__init__()
        if 'AWS_ACCESS_KEY_ID' in os.environ:
            s3 = s3fs.S3FileSystem(key=os.environ['AWS_ACCESS_KEY_ID'], secret=os.environ['AWS_SECRET_ACCESS_KEY'])
        else:
            s3 = s3fs.S3FileSystem(anon=True)
        self.files = {file: s3.open(file, 'rb') for file in file_names}
        self.token_dtype = token_dtype
        self.block_size = block_size
        self.file_sizes = {file: file_sizes[i] for i, file in enumerate(file_names)}
        self.rx_queue = rx_queue
        self.tx_queue = tx_queue

    def _read_next_block(self, file_name: str, rand_seek: int) -> Optional[np.ndarray]:
        dtype_bytes = np.dtype(self.token_dtype).itemsize
        file = self.files[file_name]
        file_size = self.file_sizes[file_name]

        if file_size < (self.block_size + 1) * dtype_bytes:
            return None

        if rand_seek:
            idx = torch.randint(low=0, high=file_size - (self.block_size + 1) * dtype_bytes - 1, size=(1,)).item()
            if idx % dtype_bytes != 0:
                idx -= idx % dtype_bytes
            if idx < 0:
                idx = 0
            file.seek(idx)

        block = np.frombuffer(file.read((self.block_size + 1) * dtype_bytes), dtype=self.token_dtype)
        return block

    def run(self) -> None:
        while True:
            file_name = self.rx_queue.get()  # wait for a signal to start reading
            block = self._read_next_block(file_name, True)
            if block is None:
                continue
            self.tx_queue.put(block)


class S3FolderDataset(Dataset):

    def __init__(self, folder_path: str, group_filter_prob_supplier: Callable[[str], Tuple[any, bool, float]],
                 seq_len: int,
                 token_dtype: np.dtype) -> None:
        self.folder_path = folder_path
        self.group_filter_prob_supplier = group_filter_prob_supplier
        self.seq_len = seq_len
        self.token_dtype = token_dtype

    def __iter__(self):
        s3 = s3fs.S3FileSystem(anon=True)

        files = []
        file_sizes = []
        unnorm_probs = []
        groups = []

        files_for_proc = []
        files_per_proc = 64
        proc_for_file = {}
        queues_for_file = {}

        def _flush():
            nonlocal files_for_proc, file_sizes
            rx_queue_ = multiprocessing.Queue()
            tx_queue_ = multiprocessing.Queue()
            proc = S3AsyncReader(files_for_proc, file_sizes, self.token_dtype, self.seq_len, rx_queue_, tx_queue_)
            proc.daemon = True
            proc.start()
            for f in files_for_proc:
                proc_for_file[f] = proc
                queues_for_file[f] = {"rx": tx_queue_, "tx": rx_queue_}
            files_for_proc = []
            file_sizes = []

        for file_name in tqdm(s3.ls(self.folder_path), desc="Listing s3 files..."):
            group, should_read, sampling_probability = self.group_filter_prob_supplier(file_name)
            if should_read:
                files_for_proc.append(file_name)
                files.append(file_name)
                file_size = s3.du(file_name)
                file_sizes.append(file_size)
                groups.append(group)

                unnorm_probs.append(sampling_probability * file_size)

                if len(files_for_proc) == files_per_proc:
                    _flush()
        _flush()

        # Account for number of files in each group
        # Together with prob being proportional to file size, this should give us a uniform distribution
        # over all tokens in the dataset
        group_counts = Counter(groups)
        for i, file in enumerate(files):
            unnorm_probs[i] /= group_counts[groups[i]]

        # Normalize probabilities
        probs = torch.tensor(unnorm_probs)
        probs /= probs.sum()

        expecting_queues = set()

        num_blocks_in_flight = 64

        while True:
            # poll expecting_queues
            to_remove = set()
            for rx_queue in expecting_queues:
                try:
                    block = rx_queue.get(block=False)
                    yield iter([(torch.from_numpy(block[:-1].astype(np.int64)),
                                 torch.from_numpy(block[1:].astype(np.int64)))])

                    if rx_queue.empty():
                        to_remove.add(rx_queue)
                except queue.Empty:
                    pass
            expecting_queues -= to_remove

            if len(expecting_queues) >= num_blocks_in_flight:
                time.sleep(0.01)
                continue

            # populate expecting_queues
            idx = torch.multinomial(probs, 1).item()

            file = files[idx]

            rx_queue = queues_for_file[file]["rx"]
            tx_queue = queues_for_file[file]["tx"]

            tx_queue.put(file)
            expecting_queues.add(rx_queue)
