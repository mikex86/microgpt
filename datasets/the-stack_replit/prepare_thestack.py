import multiprocessing
import os

import numpy as np
import pyarrow.parquet as pq
import smart_open

from tqdm import tqdm

from robustdatasets.parquet_streamer import ParquetStreamer
from tokenization.sentencepiece_tokenizer import SentencePieceTokenizer

# Specify the URL of the Parquet file
parquet_urls = [
    f"https://huggingface.co/datasets/bigcode/the-stack-dedup/resolve/main/data/python/data-00{i:03d}-of-00144.parquet"
    for i in range(0, 144)
]

TOKEN_BUFFER_SIZE = 10000


def _flush_token_buffer(token_buffer, out_file):
    token_np_array = np.array(token_buffer, dtype=np.uint16)
    token_np_array.tofile(out_file)
    out_file.flush()
    token_buffer.clear()


def process_parquet_url(parquet_url: str, i: int):
    file_path = f"train-{i}.bin"
    if os.path.exists(file_path):
        print(f"Skipping {file_path} because it already exists")
        return

    tokenizer = SentencePieceTokenizer("replit_tokenizer.model")
    streamer = ParquetStreamer(
        parquet_url,
        headers={'Authorization': 'Bearer ' + os.environ['HUGGINGFACE_TOKEN']},
        observed_rows=['content']
    )
    with open(file_path, "wb") as out_file:
        token_buffer = []
        for row in streamer:
            content = row['content']
            tokens = tokenizer.encode(content)
            token_buffer.extend(tokens)

            if len(token_buffer) >= TOKEN_BUFFER_SIZE:
                _flush_token_buffer(token_buffer, out_file)


BUFFER_SIZE = 65536 * 16


def main():
    # multiprocessing
    num_workers = 8
    with multiprocessing.Pool(num_workers) as pool:
        pool.starmap(process_parquet_url, zip(parquet_urls, range(len(parquet_urls))))
        pool.close()
        pool.join()

    # merge files into train.bin and val.bin
    with open("train.bin", "wb") as out_file:
        for i in range(0, 143):
            with open(f"train-{i}.bin", "rb") as in_file:
                while True:
                    data = in_file.read(BUFFER_SIZE)
                    if not data:
                        break
                    out_file.write(data)
            os.remove(f"train-{i}.bin")

    # rename the last file to val.bin
    os.rename(f"train-143.bin", "val.bin")


if __name__ == '__main__':
    main()
