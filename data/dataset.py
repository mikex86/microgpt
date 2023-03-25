import numpy as np
import torch


class BinaryTokenDataset:

    def __init__(self, file_path: str, seq_len: int, token_dtype: np.dtype) -> None:
        self.array = np.memmap(file_path, mode="r", dtype=token_dtype)
        self.seq_len = seq_len

    def __iter__(self):
        while True:
            idx = torch.randint(low=0, high=len(self.array) - self.seq_len - 1, size=(1,)).item()
            yield torch.from_numpy(self.array[idx:idx + self.seq_len].astype(np.int64)), \
                torch.from_numpy(self.array[idx + 1:idx + self.seq_len + 1].astype(np.int64))
