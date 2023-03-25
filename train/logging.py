import os

import wandb
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from train.checkpointing import CheckpointInfo

LOG_TO_CONSOLE = True
LOG_TO_WANDB = os.getenv("LOG_WANDB") is not None

if LOG_TO_WANDB:
    wandb.init(project="tinygpt", entity="mikex86")


def log_train_step(step: int, step_ms: float, data_dict: dict):
    if LOG_TO_CONSOLE:
        print(f"Step {step} took {step_ms:.3f}ms: {data_dict}")

    if LOG_TO_WANDB:
        data_dict["step"] = step
        data_dict["step_ms"] = step_ms
        wandb.log(data_dict)


def log_eval_step(step: int, data_dict: dict):
    eval_loss = data_dict["loss/val"]
    if LOG_TO_CONSOLE:
        print(f"Eval after step {step}: loss/val = {eval_loss}")

    if LOG_TO_WANDB:
        data_dict["step"] = step
        wandb.log(data_dict)


def log_save_checkpoint(checkpoint_info: 'CheckpointInfo'):
    if LOG_TO_CONSOLE:
        print(f"Saved checkpoint at step {checkpoint_info.step} with loss/val = {checkpoint_info.val_loss:.4f}")
