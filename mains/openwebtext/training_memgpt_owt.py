import numpy as np
import torch

from data.dataset import BinaryTokenDataset
from datasethelpers import owtdataset
from models.memgpt import MemGptModel, MemGptConfig
from train.logging import set_log_project_name
from train.training import TrainingConfig, LanguageModelTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    set_log_project_name("memgpt2-owt")

    owtdataset.download_dataset()

    memgpt_config = MemGptConfig(
        block_size=512,
        n_windows=8,
        n_layers=12,
        n_heads=12,
        n_embd=768,
        device=device,
        dtype=torch.float32,
    )

    train_ds = BinaryTokenDataset(
        "datasets/openwebtext_gpt2/train.bin",
        memgpt_config.block_size,
        np.dtype(np.uint16),
    )
    val_ds = BinaryTokenDataset(
        "datasets/openwebtext_gpt2/val.bin",
        memgpt_config.block_size,
        np.dtype(np.uint16),
    )

    training_config = TrainingConfig(
        train_dataset_iterator=iter(train_ds),
        val_dataset_iterator=iter(val_ds),

        batch_size=2,
        n_mini_steps=1,

        min_learning_rate=1e-7,
        max_learning_rate=1e-5,
        warmup_steps=100,
        max_steps=1000000,

        grad_clip=0.01,

        weight_decay=1e-1,
        betas=(0.9, 0.999),

        device=device,
        dtype=torch.float16,

        evaluation_period=50,
        num_evaluation_steps=12,

        checkpoint_dir_path="checkpoints/owt/memgpt_checkpoints",

        hyper_save_memory=False
    )
    model = MemGptModel(memgpt_config)
    trainer = LanguageModelTrainer(model, training_config)
    trainer.train()


if __name__ == '__main__':
    main()
