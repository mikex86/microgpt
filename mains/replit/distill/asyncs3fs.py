import multiprocessing
import s3fs


class AsyncS3fsWriter(multiprocessing.Process):

    def __init__(self,
                 fs: s3fs.S3FileSystem,
                 queue: multiprocessing.Queue):
        super().__init__()
        self.fs = fs
        self.rx_queue = queue
        self.file_cache = {}

    def run(self) -> None:
        while True:
            signal = self.rx_queue.get()
            if signal is None:
                return
            file_name, data = signal
            if file_name not in self.file_cache:
                self.file_cache[file_name] = self.fs.open(file_name, 'wb')
            fp = self.file_cache[file_name]
            fp.write(data)
            fp.flush()
