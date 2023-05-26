import os

import torch
import numpy as np

from datasethelpers import owtdataset
from models.replit import ReplitLMConfig, ReplitLM
from train.logging import set_log_project_name
from train.training import TrainingConfig, LanguageModelTrainer
from data.dataset import BinaryTokenDataset

from lora import lora

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    set_log_project_name("replit-tstk")

    config = ReplitLMConfig(
        d_model=2560,
        n_heads=32,
        n_layers=32,
        mlp_ratio=4,
        max_seq_len=2048,
        vocab_size=32768,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        emb_pdrop=0.0,
        alibi_bias_max=8,
        use_bias=False,
        device=device,
        dtype=torch.bfloat16
    )

    train_ds = BinaryTokenDataset(
        "datasets/the-stack_replit/train.bin",
        config.max_seq_len,
        np.dtype(np.uint16)
    )
    val_ds = BinaryTokenDataset(
        "datasets/the-stack_replit/val.bin",
        config.max_seq_len,
        np.dtype(np.uint16)
    )

    training_config = TrainingConfig(
        train_dataset_iterator=iter(train_ds),
        val_dataset_iterator=iter(val_ds),

        batch_size=1,
        n_mini_steps=3,

        min_learning_rate=6e-6,
        max_learning_rate=6e-6,
        warmup_steps=1000,
        max_steps=600000,

        grad_clip=0.01,

        weight_decay=1e-1,
        betas=(0.9, 0.999),

        device=device,
        dtype=torch.float16,

        evaluation_period=50,
        num_evaluation_steps=12,

        checkpoint_dir_path="checkpoints/replit-3b",

        hyper_save_memory=False
    )
    model = ReplitLM(config)

    # lora.lorify_module(model, 1, 1, torch.bfloat16)

    trainer = LanguageModelTrainer(model, training_config)
    trainer.train()


if __name__ == '__main__':
    main()
