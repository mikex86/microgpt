import torch
import numpy as np

from models.memgpt import MemGptModel, MemGptConfig
from train.training import TrainingConfig, LanguageModelTrainer
from data.dataset import BinaryTokenDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    memgpt_config = MemGptConfig(
        vocab_size=65,
        block_size=256,
        n_windows=4,
        n_layers=6,
        n_heads=6,
        n_embd=384
    )

    train_ds = BinaryTokenDataset(
        "datasets/shakespeare_char/train.bin",
        memgpt_config.block_size * memgpt_config.n_windows,
        np.dtype(np.uint16)
    )
    val_ds = BinaryTokenDataset(
        "datasets/shakespeare_char/val.bin",
        memgpt_config.block_size * memgpt_config.n_windows,
        np.dtype(np.uint16)
    )

    training_config = TrainingConfig(
        train_dataset_iterator=iter(train_ds),
        val_dataset_iterator=iter(val_ds),

        batch_size=8,
        n_mini_steps=2,

        min_learning_rate=1e-4,
        max_learning_rate=1e-3,
        warmup_steps=100,
        max_steps=5000,

        grad_clip=0.0,

        weight_decay=1e-1,
        betas=(0.9, 0.999),

        device=device,
        dtype=torch.float16,

        evaluation_period=25,
        num_evaluation_steps=32,

        checkpoint_dir_path="checkpoints/shakespeare/memgpt_checkpoints",
    )
    model = MemGptModel(memgpt_config)
    trainer = LanguageModelTrainer(model, training_config)
    trainer.train()


if __name__ == '__main__':
    main()
