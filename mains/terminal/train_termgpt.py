import torch

from datasethelpers import owtdataset
from environments.shell.dataset.terminal_dataset_reader import TerminalDatasetReader
from models.termgpt import TerminalGptModel, TerminalGptConfig
from train.logging import set_log_project_name
from train.training import TrainingConfig, LanguageModelTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    set_log_project_name("termgpt")

    owtdataset.download_dataset()

    train_ds = TerminalDatasetReader("datasets/terminaldummyds/term_dummy_ds.bin")
    val_ds = TerminalDatasetReader("datasets/terminaldummyds/term_dummy_ds.bin")

    gpt_config = TerminalGptConfig(
        block_size=(train_ds.width + 1) * train_ds.height + 1,
        n_layers=6,
        n_heads=8,
        n_embd=64,
        device=device,
        dtype=torch.float32,
        vocab_size=256
    )

    training_config = TrainingConfig(
        train_dataset_iterator=iter(train_ds),
        val_dataset_iterator=iter(val_ds),
        batch_size=128,
        n_mini_steps=1,

        min_learning_rate=6e-6,
        max_learning_rate=6e-5,
        warmup_steps=100,
        max_steps=2000,

        grad_clip=0.0,

        weight_decay=1e-1,
        betas=(0.9, 0.999),

        device=device,
        dtype=torch.float32,

        evaluation_period=50,
        num_evaluation_steps=12,

        checkpoint_dir_path="checkpoints/termgpt/dummyds",

        hyper_save_memory=False
    )
    model = TerminalGptModel(gpt_config)
    trainer = LanguageModelTrainer(model, training_config)
    trainer.train()


if __name__ == '__main__':
    main()
