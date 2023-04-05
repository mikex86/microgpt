import os

import torch
import numpy as np

from models.gpt2 import Gpt2Model, Gpt2Config
from train.logging import set_log_project_name
from train.training import TrainingConfig, LanguageModelTrainer
from data.dataset import BinaryTokenDataset
from urllib.request import urlretrieve
import progressbar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            urlretrieve("https://micro-gpt-datasets.s3.eu-central-1.amazonaws.com/train.bin",
                    "datasets/openwebtext_gpt2/train.bin", DownloadProgressBar())
        else:
            os.system("wget https://micro-gpt-datasets.s3.eu-central-1.amazonaws.com/train.bin -O datasets/openwebtext_gpt2/train.bin")
    if not os.path.exists("datasets/openwebtext_gpt2/val.bin"):
        if os.name == 'nt':
            urlretrieve("https://micro-gpt-datasets.s3.eu-central-1.amazonaws.com/val.bin",
                    "datasets/openwebtext_gpt2/val.bin", DownloadProgressBar())
        else:
            os.system("wget https://micro-gpt-datasets.s3.eu-central-1.amazonaws.com/val.bin -O datasets/openwebtext_gpt2/val.bin")


def main():
    set_log_project_name("memgpt-owt")

    download_dataset()

    gpt_config = Gpt2Config(
        block_size=512,
        n_layers=8,
        n_heads=8,
        n_embd=1024,
        device=device,
        dtype=torch.float32,
    )

    train_ds = BinaryTokenDataset(
        "datasets/openwebtext_gpt2/train.bin",
        gpt_config.block_size,
        np.dtype(np.uint16)
    )
    val_ds = BinaryTokenDataset(
        "datasets/openwebtext_gpt2/val.bin",
        gpt_config.block_size,
        np.dtype(np.uint16)
    )

    training_config = TrainingConfig(
        train_dataset_iterator=iter(train_ds),
        val_dataset_iterator=iter(val_ds),

        batch_size=2,
        n_mini_steps=2,

        min_learning_rate=1e-7,
        max_learning_rate=1e-5,
        warmup_steps=100,
        max_steps=5000,

        grad_clip=0.001,

        weight_decay=1e-1,
        betas=(0.9, 0.999),

        device=device,
        dtype=torch.float16,

        evaluation_period=50,
        num_evaluation_steps=12,

        checkpoint_dir_path="checkpoints/owt/gpt2_checkpoints",

        hyper_save_memory=False
    )
    model = Gpt2Model(gpt_config)
    trainer = LanguageModelTrainer(model, training_config)
    trainer.train()


if __name__ == '__main__':
    main()
