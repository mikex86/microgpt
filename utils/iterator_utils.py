import torch
import queue
import threading


def make_batched_iterator(dataset_iterator: iter,
                          batch_size: int,
                          device: torch.device):
    """
    Takes an iterator over individual examples and returns an iterator over batches of examples.
    If the device is of type "cuda", the yielded batches are pinned to memory and non-blocking
    :param dataset_iterator: an infinite iterator over examples (x, y) where x and y are tensors of shape (seq_len,)
    :param batch_size: the number of examples in each batch
    :param device: the device on which to place the yielded batches on
    :return: an infinite iterator over batches of examples (x, y)
            where x and y are tensors of shape (batch_size, seq_len)
    """
    while True:
        examples_x, examples_y = [], []
        for i in range(batch_size):
            x, y = next(dataset_iterator)
            examples_x.append(x)
            examples_y.append(y)
        x, y = torch.stack(examples_x, dim=0), torch.stack(examples_y, dim=0)
        if device.type == "cuda":
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        yield x, y


def prefetching_iterator(dataset_iterator: iter,
                         num_prefetch: int):
    """
    Takes an iterator and wraps it in a background prefetching iterator
    :param dataset_iterator: an infinite iterator over examples
    :param num_prefetch: the number of examples to prefetch in the background
    :return:
    """

    # create a queue to store the prefetched examples
    prefetch_queue = queue.Queue(maxsize=num_prefetch)

    done = False

    # define the function that will be run in the background
    def prefetch():
        nonlocal done
        for example in dataset_iterator:
            prefetch_queue.put(example)
        done = True

    # start the background thread
    thread = threading.Thread(target=prefetch)
    thread.daemon = True
    thread.start()

    # yield the prefetched examples
    while not done:
        yield prefetch_queue.get()
        prefetch_queue.task_done()
