import os
import struct
import typing
from typing import Iterator, Tuple, Union
import torch

from tokenization.tokenizer import Tokenizer


class TerminalDatasetReader:

    def __init__(self, dataset_file: Union[str, typing.BinaryIO, typing.IO[bytes]],
                 tokenizer: Tokenizer,
                 shuffle: bool = True,
                 balance_action_no_action: bool = False,
                 balance_factor: float = 0.25):
        """
        :param dataset_file: the terminal recording file to read from
        :param tokenizer: the tokenizer to decode the completion bytes with
        :param shuffle: Whether to shuffle the terminal-input pairs. If false, temporal order is preserved.
        :param balance_action_no_action: balances the number of action and no action samples. Only applies if shuffle is True.
        :param balance_factor: the ratio of no action to action samples. Only applies if shuffle is True.
        """
        if isinstance(dataset_file, str):
            self.dataset_file = open(dataset_file, 'rb')
        else:
            self.dataset_file = dataset_file
        self.tokenizer = tokenizer
        self.balance_action_no_action = balance_action_no_action
        self.shuffle = shuffle
        self.balance_factor = balance_factor
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
                self.dataset_file.seek((4 + 4) + (rand_idx * ((self.width + 1) * self.height + 4)))
            try:
                target = self._read_target()

                # balance no action and action
                if self.shuffle and self.balance_action_no_action and target == 0 and n_no_action / self.balance_factor > n_action:
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
        read = self.dataset_file.read(4)

        if len(read) == 0:
            raise EOFError()

        read = read.decode('ISO-8859-1', 'error')

        # strip leading \x00 except for the last one
        if read == '\x00\x00\x00\x00':
            read = '\x00'  # preserve the no action token
        else:
            read = read.rstrip('\x00')  # interpret \x00 as padding

        tokens = self.tokenizer.encode(read)

        if len(tokens) != 1:
            raise ValueError(f'Completion cannot tokenize to more than one token!')

        target[0] = int(tokens[0])
        return target

    def _count_samples(self) -> int:
        self.dataset_file.seek(0, os.SEEK_END)
        n_bytes = self.dataset_file.tell()
        # skip header
        n_bytes -= 4 + 4
        # each sample is (width + 1) * height + 1 bytes
        n_samples = n_bytes // ((self.width + 1) * self.height + 1)
        return n_samples
