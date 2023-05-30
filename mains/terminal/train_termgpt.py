import torch

from datasethelpers import owtdataset
from environments.shell.dataset.terminal_task_dataset import TerminalTaskDataset
from models.termgpt import TerminalGptModel, TerminalGptConfig
from tokenization.greedy_tokenizer import GreedyTokenizer
from train import logging
from train.logging import set_log_project_name
from train.training import TrainingConfig, LanguageModelTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64


def main():
    set_log_project_name("termgpt")

    owtdataset.download_dataset()

    tokenizer = GreedyTokenizer.from_json("datasets/codeds/tokenizer.json")

    train_ds = TerminalTaskDataset("datasets/codeds/leetcode.zip", tokenizer, batch_size=BATCH_SIZE)
    val_ds = TerminalTaskDataset("datasets/codeds/leetcode.zip", tokenizer, batch_size=BATCH_SIZE)

    gpt_config = TerminalGptConfig(
        block_size=(train_ds.width + 1) * train_ds.height + 1,
        n_layers=12,
        n_action_detection_layers=2,
        n_heads=8,
        n_embd=64,
        device=device,
        dtype=torch.float32,
        vocab_size=tokenizer.vocab_size,
        dropout=0.0
    )

    total_loss_no_action = 0.0
    total_loss_action = 0.0

    n_ministeps = 3

    def on_mini_step(
            step: int, mini_step: int,
            _ce_loss: float,
            _kl_loss: float,
            _x: torch.tensor,  # (batch_size, block_size)
            y: torch.tensor,  # (batch_size, 1)
            logits: torch.tensor  # (batch_size, vocab_size)
    ):
        nonlocal total_loss_no_action, total_loss_action
        if mini_step == n_ministeps - 1:
            logging.log_train_step_extra(step, {
                "loss/train_no_action": total_loss_no_action / n_ministeps,
                "loss/train_action": total_loss_action / n_ministeps
            })
            total_loss_no_action = 0
            total_loss_action = 0

        no_action_idx, _ = torch.where(y == 0)
        action_idx, _ = torch.where(y != 0)

        no_action_y = y[no_action_idx]
        action_y = y[action_idx]

        no_action_logits = logits[no_action_idx]
        action_logits = logits[action_idx]

        no_action_loss = torch.nn.functional.cross_entropy(no_action_logits.view(-1, logits.size(-1)),
                                                           no_action_y.view(-1))

        total_loss_no_action += no_action_loss.item()

        action_loss = torch.nn.functional.cross_entropy(action_logits.view(-1, logits.size(-1)), action_y.view(-1))
        total_loss_action += action_loss.item()

    training_config = TrainingConfig(
        train_dataset_iterator=iter(train_ds),
        val_dataset_iterator=iter(val_ds),
        batch_size=BATCH_SIZE,
        n_mini_steps=n_ministeps,

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

        checkpoint_dir_path="checkpoints/termgpt/codeds",

        hyper_save_memory=False,

        mini_step_listener=on_mini_step
    )

    model = TerminalGptModel(gpt_config)
    trainer = LanguageModelTrainer(model, training_config)
    trainer.train()


if __name__ == '__main__':
    main()
