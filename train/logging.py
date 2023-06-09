import os

import wandb
from typing import TYPE_CHECKING
import traceback

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


def log_train_step_extra(step: int, data_dict: dict):
    __init_wandb()
    if LOG_TO_CONSOLE:
        print(f"Step {step}: {data_dict}")

    if LOG_TO_WANDB:
        data_dict["step"] = step
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
    __init_wandb()
    if LOG_TO_CONSOLE:
        print(
            f"Saved checkpoint at step {checkpoint_info.step}"
            f" with loss/val = {checkpoint_info.val_loss:.4f} blocked main thread for {saving_time_seconds:.3f}s"
        )


def log_loss_nan(current_step: int):
    __init_wandb()
    print(f"WARNING: Loss is NaN at step {current_step}. Attempting to recover...")

    if LOG_TO_WANDB:
        wandb.alert(title="Loss is NaN", text=f"Loss is NaN at step {current_step}!")


def log_tried_save_nan_checkpoint(current_step: int):
    __init_wandb()
    print(f"WARNING: Tried to save checkpoint with NaN loss at step {current_step}!")

    if LOG_TO_WANDB:
        wandb.alert(title="Tried to save NaN checkpoint",
                    text=f"Tried to save checkpoint with NaN loss at step {current_step}!")


def log_error(e: BaseException):
    __init_wandb()
    print(f"ERROR: {e}")
    traceback.print_exc()
    if LOG_TO_WANDB:
        wandb.alert(title="Error", text=str(e) + "\n" + traceback.format_exc())


def log_oom(step: int):
    __init_wandb()
    print(f"ERROR: Out of memory at step {step}!")
    if LOG_TO_WANDB:
        wandb.alert(title="Out of memory", text=f"Out of memory at step {step}!")


def log_blocking_save(checkpoint_dir_path: str):
    __init_wandb()
    print(
        f"Blocking save of checkpoint to {checkpoint_dir_path}! Previous checkpoint hasn't finished saving yet!")


def log_async_save_start(save_id, checkpoint_dir_path: str):
    __init_wandb()
    print(f"Starting async save {save_id} to {checkpoint_dir_path}...")


def log_async_save_end(save_id, checkpoint_dir_path: str, saving_time_seconds: float):
    __init_wandb()
    print(f"Finished async save {save_id} to {checkpoint_dir_path} in {saving_time_seconds:.3f}s.")


def log_waiting_for_upload(save_id, checkpoint_dir_path: str):
    __init_wandb()
    print(f"Waiting for async save {save_id} to {checkpoint_dir_path} to upload...")
