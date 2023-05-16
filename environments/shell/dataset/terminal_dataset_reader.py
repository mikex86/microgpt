import struct
from typing import Iterator, Tuple
import torch

from data.dataset import Dataset


class TerminalDatasetReader(Dataset):

    def __init__(self, file_path: str):
        self.dataset_file = open(file_path, 'rb')
        self.width, self.height = self._read_header()

    def _read_header(self):
        width = struct.unpack('>i', self.dataset_file.read(4))[0]
        height = struct.unpack('>i', self.dataset_file.read(4))[0]
        return width, height

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        while True:
            try:
                context = self._read_context()
                target = self._read_target()
                yield context, target
            except EOFError:
                self.dataset_file.seek(4 + 4)  # skip header

    def _read_context(self) -> torch.Tensor:
        # read bytes (1 byte corresponds to one iso-8859-1 character)
        read = self.dataset_file.read((self.width + 1) * self.height)
        if len(read) < (self.width + 1) * self.height:
            raise EOFError()

        # convert to tensor
        context = torch.zeros((self.width + 1) * self.height, dtype=torch.int64)
        for i in range(len(read)):
            context[i] = int(read[i])
        return context

    def _read_target(self) -> torch.Tensor:
        target = torch.zeros(1, dtype=torch.int64)
        read = self.dataset_file.read(1)
        if len(read) == 0:
            raise EOFError()
        target[0] = int(read[0])
        return target
