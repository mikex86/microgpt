from typing import Iterator, Tuple

import torch
import queue
import threading


def make_batched_iterator(dataset_iterator: Iterator[Iterator[Tuple[torch.Tensor, torch.Tensor]]],
                          batch_size: int,
                          device: torch.device):
    """
    Takes an iterator over individual examples and returns an iterator over batches of examples.
    If the device is of type "cuda", the yielded batches are pinned to memory and non-blocking
    :param dataset_iterator: an infinite iterator over iterators of examples (x, y) where x and y are tensors of shape (seq_len,).
    This subiterator of examples must be processed in order as to not distort the dataset.
    :param batch_size: the number of examples in each batch
    :param device: the device on which to place the yielded batches on
    :return: an infinite iterator over batches of examples (x, y)
            where x and y are tensors of shape (batch_size, seq_len)
    """
    while True:
        sequence_it = next(dataset_iterator)
        examples_x, examples_y = [], []
        for i in range(batch_size):
            try:
                x, y = next(sequence_it)
            except StopIteration:
                sequence_it = next(dataset_iterator)
                x, y = next(sequence_it)
            examples_x.append(x)
            examples_y.append(y)
        x, y = torch.stack(examples_x, dim=0), torch.stack(examples_y, dim=0)
        if device.type == "cuda":
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        yield x, y


def prefetching_iterator(dataset_iterator: iter, num_prefetch: int):
    """
    Takes an iterator and wraps it in a background prefetching iterator.
    Must be wrapped with a "with" statement to ensure the background thread is properly terminated
    and the prefetching queue is properly freed.
    :param dataset_iterator: an infinite iterator over examples
    :param num_prefetch: the number of examples to prefetch in the background
    :return:
    """

    class PrefetchingIterator:
        def __init__(self, dataset_iterator, num_prefetch):
            self.prefetch_queue = queue.Queue(maxsize=num_prefetch)
            self.done = False
            self.dataset_iterator = dataset_iterator

        def __enter__(self):
            self.thread = threading.Thread(target=self.prefetch)
            self.thread.daemon = True
            self.thread.start()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.done = True
            self.thread.join()

        def __iter__(self):
            return self

        def __next__(self):
            if not self.done:
                return self.prefetch_queue.get()
            else:
                raise StopIteration

        def prefetch(self):
            for example in self.dataset_iterator:
                if self.done:
                    break
                self.prefetch_queue.put(example)

    return PrefetchingIterator(dataset_iterator, num_prefetch)
