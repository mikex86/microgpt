import io
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset  # huggingface datasets
import s3fs
from utils import iterator_utils

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# Configure the following placeholders
s3_bucket = 'micro-gpt-datasets-us'
s3_prefix = 'the-pile'

dataset = load_dataset("EleutherAI/the_pile", streaming=True)

# we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
enc = tiktoken.get_encoding("gpt2")


def process(example):
    ids = enc.encode_ordinary(example['text'])  # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {'ids': ids, 'len': len(ids)}
    return out


# tokenize the dataset
tokenized = dataset.map(
    process,
    remove_columns=['text']
)

s3 = s3fs.S3FileSystem()

# concatenate all the ids in each dataset into one large file we can use for training
for split, split_dset in tokenized.items():

    # one file for each split
    s3_key = f"{s3_bucket}/{s3_prefix}/{split}.bin"
    dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)

    print(f"writing {s3_key} to s3://{s3_bucket}...")

    # Create a buffer to hold the chunk of data
    chunk_size = 1024 * 1024 * 1024  # 1GB chunks
    buffer = io.BytesIO()

    # check if the file already exists
    if s3.exists(s3_key):
        print(f"file {s3_key} already exists. skipping...")
        continue

    s3.touch(s3_key, create_parents=True)
    with s3.open(s3_key, 'wb') as f:
        idx = 0
        with iterator_utils.prefetching_iterator(split_dset, num_prefetch=10000) as it:
            for example in tqdm(it):
                # Write the current example to the buffer
                arr = np.array(example['ids'], dtype=dtype).tobytes()
                f.write(arr)

                idx += example['len']
    print(f"done writing split {split}.")
