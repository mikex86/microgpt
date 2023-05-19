import os
import struct
from typing import Iterator, Tuple
import torch

from data.dataset import Dataset


class TerminalDatasetReader(Dataset):

    def __init__(self, file_path: str, shuffle: bool = False,
                 balance_action_no_action: bool = True,
                 balance_factor: float = 0.25):
        self.balance_action_no_action = balance_action_no_action
        self.shuffle = shuffle
        self.balance_factor = balance_factor
        self.dataset_file = open(file_path, 'rb')
        self.width, self.height = self._read_header()
        self.n_samples = self._count_samples()
        self.dataset_file.seek(4 + 4)  # skip header

    def _read_header(self):
        width = struct.unpack('>i', self.dataset_file.read(4))[0]
        height = struct.unpack('>i', self.dataset_file.read(4))[0]
        return width, height

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        n_no_action = 0
        n_action = 0
        while True:
            if self.shuffle:
                rand_idx = torch.randint(0, self.n_samples, (1,)).item()
                self.dataset_file.seek((4 + 4) + (rand_idx * ((self.width + 1) * self.height + 1)))
            try:
                target = self._read_target()

                if self.balance_action_no_action and target == 0 and n_no_action / self.balance_factor > n_action:  # balance no action and action
                    continue

                context = self._read_context()
                if target == 0:
                    n_no_action += 1
                else:
                    n_action += 1
                yield context, target
            except EOFError:
                self.dataset_file.seek(4 + 4)  # skip header

    def _read_context(self) -> torch.Tensor:
        # read bytes (1 byte corresponds to one iso-8859-1 character)
        read = self.dataset_file.read((self.width + 1) * self.height)
        if len(read) < (self.width + 1) * self.height:
            raise EOFError()

        # convert to tensor
        return torch.tensor(list(read), dtype=torch.int64)

    def _read_target(self) -> torch.Tensor:
        target = torch.zeros(1, dtype=torch.int64)
        read = self.dataset_file.read(1)
        if len(read) == 0:
            raise EOFError()
        target[0] = int(read[0])
        return target

    def _count_samples(self) -> int:
        self.dataset_file.seek(0, os.SEEK_END)
        n_bytes = self.dataset_file.tell()
        # skip header
        n_bytes -= 4 + 4
        # each sample is (width + 1) * height + 1 bytes
        n_samples = n_bytes // ((self.width + 1) * self.height + 1)
        return n_samples
