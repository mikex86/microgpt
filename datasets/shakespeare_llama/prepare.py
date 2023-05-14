import os

import numpy as np

from models.llama import LlamaTokenizer

if __name__ == '__main__':
    ds_dir = "datasets/shakespeare_llama"

    f = open(os.path.join(ds_dir, "tiny_shakespeare.txt"), "r")
    text = f.read()

    tokenizer = LlamaTokenizer(model_path='checkpoints/llama/tokenizer.model')

    train_tokens = tokenizer.encode(text[:int(len(text) * 0.8)], bos=False, eos=False)
    val_tokens = tokenizer.encode(text[int(len(text) * 0.8):], bos=False, eos=False)

    arr = np.memmap(os.path.join(ds_dir, "train.bin"), dtype=np.float16, mode='w+', shape=(len(train_tokens),))
    arr[:] = train_tokens[:]
    del arr

    arr = np.memmap(os.path.join(ds_dir, "val.bin"), dtype=np.float16, mode='w+', shape=(len(val_tokens),))
    arr[:] = val_tokens[:]
    del arr
