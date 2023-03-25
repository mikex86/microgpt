import torch
import numpy as np

from models.gpt2 import Gpt2Model, Gpt2Config
from train.training import TrainingConfig, LanguageModelTrainer
from data.dataset import BinaryTokenDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    gpt2config = Gpt2Config(
        vocab_size=65,
        block_size=256,
        n_layers=6,
        n_heads=6,
        n_embd=384
    )

    train_ds = BinaryTokenDataset(
        "datasets/shakespeare_char/train.bin",
        gpt2config.block_size,
        np.dtype(np.uint16)
    )
    val_ds = BinaryTokenDataset(
        "datasets/shakespeare_char/val.bin",
        gpt2config.block_size,
        np.dtype(np.uint16)
    )

    training_config = TrainingConfig(
        train_dataset_iterator=iter(train_ds),
        val_dataset_iterator=iter(val_ds),

        batch_size=64,
        n_mini_steps=5,

        learning_rate=1e-4,
        weight_decay=1e-1,
        betas=(0.9, 0.999),

        device=device,
        dtype=torch.float16,

        evaluation_period=200,
        num_evaluation_steps=32,

        checkpoint_dir_path="checkpoints",
    )
    model = Gpt2Model(gpt2config)
    trainer = LanguageModelTrainer(model, training_config)
    trainer.train()


if __name__ == '__main__':
    main()
