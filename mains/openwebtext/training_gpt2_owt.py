import os

import torch
import numpy as np

from datasethelpers import owtdataset
from models.gpt2 import Gpt2Model, Gpt2Config
from train.logging import set_log_project_name
from train.training import TrainingConfig, LanguageModelTrainer
from data.dataset import BinaryTokenDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    set_log_project_name("memgpt-owt")

    owtdataset.download_dataset()

    gpt_config = Gpt2Config(
        block_size=1024,
        n_layers=12,
        n_heads=12,
        n_embd=768,
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

        batch_size=3,
        n_mini_steps=5,

        min_learning_rate=6e-6,
        max_learning_rate=6e-4,
        warmup_steps=100,
        max_steps=5000,

        grad_clip=1.0,

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
