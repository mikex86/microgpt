import os

import wandb
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from train.checkpointing import CheckpointInfo

LOG_PROJECT_NAME = "lmtest"


def set_log_project_name(name: str):
    global LOG_PROJECT_NAME
    LOG_PROJECT_NAME = name


LOG_TO_CONSOLE = True
LOG_TO_WANDB = os.getenv("LOG_WANDB") == "1"
__initialized = False


def __init_wandb():
    if not LOG_TO_WANDB:
        return
    global __initialized
    if not __initialized:
        wandb.init(project=LOG_PROJECT_NAME)
        __initialized = True


def log_train_step(step: int, step_ms: float, data_dict: dict):
    __init_wandb()
    if LOG_TO_CONSOLE:
        print(f"Step {step} took {step_ms:.3f}ms: {data_dict}")

    if LOG_TO_WANDB:
        data_dict["step"] = step
        data_dict["step_ms"] = step_ms
        wandb.log(data_dict)


def log_eval_step(step: int, data_dict: dict):
    __init_wandb()
    eval_loss = data_dict["loss/val"]
    if LOG_TO_CONSOLE:
        print(f"Eval after step {step}: loss/val = {eval_loss}")

    if LOG_TO_WANDB:
        data_dict["step"] = step
        wandb.log(data_dict)


def log_save_checkpoint(checkpoint_info: 'CheckpointInfo', saving_time_seconds: float):
    if LOG_TO_CONSOLE:
        print(
            f"Saved checkpoint at step {checkpoint_info.step}"
            f" with loss/val = {checkpoint_info.val_loss:.4f} took {saving_time_seconds:.3f}s"
        )


def log_loss_nan(current_step: int):
    print(f"WARNING: Loss is NaN at step {current_step}!")


def log_tried_save_nan_checkpoint(current_step: int):
    print(f"WARNING: Tried to save checkpoint with NaN loss at step {current_step}!")
