from dataclasses import dataclass
import os
import json
from typing import Optional

import torch
from torch.nn import Module, Parameter
from torch.optim import Optimizer


@dataclass
class CheckpointInfo:
    val_loss: float
    step: int


def _get_checkpoint_info(checkpoint_path: str) -> Optional[CheckpointInfo]:
    checkpoint_info_file_path = os.path.join(checkpoint_path, "checkpoint_info.json")
    if not os.path.exists(checkpoint_info_file_path):
        return None
    with open(checkpoint_info_file_path, "r") as f:
        checkpoint_info_dict = json.load(f)
    return CheckpointInfo(**checkpoint_info_dict)


def get_checkpoint_info(checkpoint_path: str, checkpoint_name: str) -> CheckpointInfo:
    return _get_checkpoint_info(os.path.join(checkpoint_path, checkpoint_name))


def _save_checkpoint(model: Module, optimizer: Optimizer, checkpoint_dir_path: str, checkpoint_info: CheckpointInfo):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()  # make a copy of the optimizer state

    os.makedirs(checkpoint_dir_path, exist_ok=True)
    checkpoint_info_file_path = os.path.join(checkpoint_dir_path, "checkpoint_info.json")
    with open(checkpoint_info_file_path, "w") as f:
        json.dump(checkpoint_info.__dict__, f)

    checkpoint_file = os.path.join(checkpoint_dir_path, "checkpoint.pt")
    torch.save({
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state
    }, checkpoint_file)


def save_checkpoint(model: Module, optimizer: Optimizer, checkpoint_dir_path: str, checkpoint_name: str,
                    checkpoint_info: CheckpointInfo):
    _save_checkpoint(model, optimizer, os.path.join(checkpoint_dir_path, checkpoint_name), checkpoint_info)


def _load_checkpoint(model: Module, optimizer: Optional[Optimizer], checkpoint_dir_path: str):
    checkpoint_file = os.path.join(checkpoint_dir_path, "checkpoint.pt")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    del checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_checkpoint(model: Module, optimizer: Optional[Optimizer], checkpoint_dir_path: str, checkpoint_name: str):
    _load_checkpoint(model, optimizer, os.path.join(checkpoint_dir_path, checkpoint_name))
