import multiprocessing
import os
import queue
import threading
from typing import Mapping, Tuple, Iterator, List, Optional

import numpy as np
import s3fs as s3fs

N_SEQUENTIAL_BLOCKS = 10
SEQUENTIAL_SKIP = 1000

BUFFER_SIZE = 2 ** 20


class BlockStreamingProcess(multiprocessing.Process):

    def __init__(self, file_names: List[str], token_dtype: np.dtype, block_size: int,
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
        self.rx_queue = rx_queue
        self.tx_queue = tx_queue
        self.buffer = bytearray(BUFFER_SIZE)
        self.buffer_read_pos = 0
        self.buffer_write_pos = 0

    def __buffered_read(self, file, n_bytes) -> Optional[bytearray]:
        bytes_read = 0
        while bytes_read < n_bytes:
            if self.buffer_read_pos == self.buffer_write_pos:
                # buffer is empty, read more data
                self.buffer_read_pos = 0
                prev_buffer_write_pos = self.buffer_write_pos
                self.buffer_write_pos = file.readinto(self.buffer)
                if prev_buffer_write_pos == 0 and self.buffer_write_pos == 0:
                    return None
                if self.buffer_write_pos == 0:
                    file.seek(0)
                    continue
            bytes_to_read = min(n_bytes - bytes_read, self.buffer_write_pos - self.buffer_read_pos)
            self.buffer_read_pos += bytes_to_read
            bytes_read += bytes_to_read
        return self.buffer[self.buffer_read_pos - bytes_read:self.buffer_read_pos]

    def _read_next_block(self, file_name: str) -> Optional[np.ndarray]:
        dtype_bytes = np.dtype(self.token_dtype).itemsize
        file = self.files[file_name]
        block = self.__buffered_read(file, self.block_size * dtype_bytes)
        if block is None:
            return None
        block = np.frombuffer(block, dtype=self.token_dtype).astype(np.int64)
        return block

    def run(self) -> None:
        while True:
            file_name = self.rx_queue.get()  # wait for a signal to start reading
            block = None
            while block is None:
                block = self._read_next_block(file_name)

            self.tx_queue.put(block)


torch_mod = None


def get_block_iters(dataset_s3_folder: str,
                    language_probabilities: Mapping[str, float],
                    num_blocks_in_flight: int,
                    block_size: int,
                    train_val_probs: Tuple[float, float],
                    token_dtype: np.dtype) -> Tuple[Iterator[np.ndarray], Iterator[np.ndarray]]:
    """
    :return: train_it, val_it
    """
    global torch_mod

    # check if torch has already been imported
    if torch_mod is None:
        import torch
        torch_mod = torch_mod

    if 'AWS_ACCESS_KEY_ID' in os.environ:
        s3 = s3fs.S3FileSystem(key=os.environ['AWS_ACCESS_KEY_ID'], secret=os.environ['AWS_SECRET_ACCESS_KEY'])
    else:
        s3 = s3fs.S3FileSystem(anon=True)

    # Create block streaming processes for all files
    files = s3.ls(dataset_s3_folder)

    languages = language_probabilities.keys()

    # get all files for each language
    train_language_files = {lang: [] for lang in languages}
    val_language_files = {lang: [] for lang in languages}
    for file in files:
        for lang in languages:
            if lang in file:
                if 'train' in file:
                    train_language_files[lang].append(file)
                elif 'val' in file:
                    val_language_files[lang].append(file)
                else:
                    raise ValueError(f"Unknown dataset split type: {file}")

    # create block streaming processes for each language file
    n_files_per_process = 128
    processes = {}
    queues = {}

    files_for_proc = []

    def __launch_proc():
        nonlocal files_for_proc
        if len(files_for_proc) == 0:
            return
        rx_queue = multiprocessing.Queue()
        tx_queue = multiprocessing.Queue()
        streaming_process = BlockStreamingProcess(files_for_proc.copy(), token_dtype, block_size + 1, rx_queue,
                                                  tx_queue)  # +1 for next token because language model 'n stuff
        streaming_process.start()
        for file_for_proc in files_for_proc:
            processes[file_for_proc] = streaming_process
            queues[file_for_proc] = {"tx_queue": rx_queue, "rx_queue": tx_queue}
        files_for_proc = []

    for lang, files in list(train_language_files.items()) + list(val_language_files.items()):
        for file in files:
            files_for_proc.append(file)

            if len(files_for_proc) >= n_files_per_process:
                __launch_proc()
    __launch_proc()

    train_blocks = queue.Queue()
    val_blocks = queue.Queue()

    def prefetch_blocks():
        processes_working_on_blocks = []
        nonlocal train_blocks, val_blocks, processes, queues
        while True:
            # Handle finished blocks
            i = 0
            while i < len(processes_working_on_blocks):
                process_entry = processes_working_on_blocks[i]
                corr_queue = process_entry["queues"]
                is_train = process_entry["is_train"]
                file_name = process_entry["file_name"]
                try:
                    block = corr_queue["rx_queue"].get(block=False)
                except queue.Empty:
                    i += 1
                    continue
                if is_train:
                    train_blocks.put({"block": block, "queues": corr_queue, "file_name": file_name})
                else:
                    val_blocks.put({"block": block, "queues": corr_queue, "file_name": file_name})
                del processes_working_on_blocks[i]
                if i > 0:
                    i -= 1

            # Handle scheduling blocks in flight
            if len(processes_working_on_blocks) >= num_blocks_in_flight:
                continue

            # get random language according to probabilities
            lang = np.random.choice(list(language_probabilities.keys()), p=list(language_probabilities.values()))

            # get random file for that language
            is_train = np.random.choice([True, False], p=train_val_probs)

            if is_train:
                files_list = train_language_files.get(lang, [])
            else:
                files_list = val_language_files.get(lang, [])

            if len(files_list) == 0:
                continue

            rand_file = np.random.choice(train_language_files[lang])

            # get corresponding process and queue
            corr_process = processes[rand_file]
            corr_queue = queues[rand_file]

            # request block from process
            corr_queue["tx_queue"].put(rand_file)

            # add process to list of processes working on blocks
            processes_working_on_blocks.append({
                "process": corr_process,
                "is_train": is_train, "queues": corr_queue,
                "file_name": rand_file
            })

    # start prefetching thread
    prefetching_thread = threading.Thread(target=prefetch_blocks, name="BlockPrefetchingThread", daemon=True)
    prefetching_thread.start()

    def block_iterator(is_train: bool) -> iter:
        if is_train:
            nonlocal train_blocks
            blocks = train_blocks
        else:
            nonlocal val_blocks
            blocks = val_blocks

        while True:
            try:
                block_entry = blocks.get(block=False)
                block = block_entry["block"]

                x = torch.from_numpy(block[:-1])
                y = torch.from_numpy(block[1:])
                yield iter([(x, y)])
            except queue.Empty:
                # print("Starved for blocks, waiting...")
                # time.sleep(0.001)
                continue

    train_it = block_iterator(True)
    val_it = block_iterator(False)

    return train_it, val_it
