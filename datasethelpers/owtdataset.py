import os
from urllib.request import urlretrieve

from progressbar import progressbar

BUCKET_URL = "https://micro-gpt-datasets-us.s3.us-west-1.amazonaws.com"


def download_dataset():
    class DownloadProgressBar:
        def __init__(self):
            self.pbar = None

        def __call__(self, block_num, block_size, total_size):
            if not self.pbar:
                self.pbar = progressbar.ProgressBar(maxval=total_size)
                self.pbar.start()

            downloaded = block_num * block_size
            if downloaded < total_size:
                self.pbar.update(downloaded)
            else:
                self.pbar.finish()

    if not os.path.exists("datasets/openwebtext_gpt2/train.bin"):
        if os.name == 'nt':
            urlretrieve(f"{BUCKET_URL}/train.bin",
                        "datasets/openwebtext_gpt2/train.bin", DownloadProgressBar())
        else:
            os.system(f"wget {BUCKET_URL}/train.bin -O datasets/openwebtext_gpt2/train.bin")
    if not os.path.exists("datasets/openwebtext_gpt2/val.bin"):
        if os.name == 'nt':
            urlretrieve(f"{BUCKET_URL}/val.bin",
                        "datasets/openwebtext_gpt2/val.bin", DownloadProgressBar())
        else:
            os.system(f"wget {BUCKET_URL}/val.bin -O datasets/openwebtext_gpt2/val.bin")
