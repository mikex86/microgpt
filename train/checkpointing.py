import copy
import multiprocessing
import time
from dataclasses import dataclass
import os
import json
from typing import Optional

import torch
from torch.nn import Module, Parameter
from torch.optim import Optimizer

from train import logging
from train.logging import log_save_checkpoint
from utils import torchhacks


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


running_save_processes = {}


def _save_checkpoint(model: Module, optimizer: Optimizer, checkpoint_dir_path: str, checkpoint_info: CheckpointInfo):
    model_state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()  # make a copy of the optimizer state

    device = next(model.parameters()).device  # hack to get the device of the model

    if device.type == 'cuda':
        copy_model_state = {k: v.cpu() for k, v in model_state_dict.items()}
        copy_optimizer_state = {k: v.cpu() for k, v in optimizer_state_dict.items()}
    else:
        copy_model_state = copy.deepcopy(model_state_dict)
        copy_optimizer_state = copy.deepcopy(optimizer_state_dict)

    # launch a separate process to save the checkpoint
    class SaveProcess(multiprocessing.Process):

        def __init__(self, save_id, model_state, optimizer_state):
            super().__init__()
            self.save_id = save_id
            self.model_state = model_state
            self.optimizer_state = optimizer_state

        def run(self):
            os.makedirs(checkpoint_dir_path, exist_ok=True)
            checkpoint_info_file_path = os.path.join(checkpoint_dir_path, "checkpoint_info.json")
            with open(checkpoint_info_file_path, "w") as f:
                json.dump(checkpoint_info.__dict__, f)

            checkpoint_file = os.path.join(checkpoint_dir_path, "checkpoint.model.pt")
            torch.save({
                "model_state_dict": self.model_state,
            }, checkpoint_file)

            checkpoint_file = os.path.join(checkpoint_dir_path, "checkpoint.optimizer.pt")
            torch.save({
                "optimizer_state_dict": self.optimizer_state,
            }, checkpoint_file)

            logging.log_async_save_end(save_id, checkpoint_dir_path)

    if checkpoint_dir_path in running_save_processes:
        # wait for the previous save process to finish
        logging.log_blocking_save(checkpoint_dir_path)
        running_save_processes[checkpoint_dir_path].join()
        del running_save_processes[checkpoint_dir_path]

    save_id = hash(time.time())
    save_process = SaveProcess(save_id, copy_model_state, copy_optimizer_state)
    save_process.start()
    logging.log_async_save_start(save_id, checkpoint_dir_path)

    running_save_processes[checkpoint_dir_path] = save_process


def save_checkpoint(model: Module, optimizer: Optimizer, checkpoint_dir_path: str, checkpoint_name: str,
                    checkpoint_info: CheckpointInfo):
    _save_checkpoint(model, optimizer, os.path.join(checkpoint_dir_path, checkpoint_name), checkpoint_info)


def _load_checkpoint(model: Module, optimizer: Optional[Optimizer], checkpoint_dir_path: str, load_lazy: bool = False):
    model_checkpoint_file = os.path.join(checkpoint_dir_path, "checkpoint.model.pt")

    if load_lazy:
        model_checkpoint = torchhacks.lazy_load(model_checkpoint_file)
    else:
        model_checkpoint = torch.load(model_checkpoint_file, map_location='cpu')
    model.load_state_dict(model_checkpoint["model_state_dict"]
                          if 'model_state_dict' in model_checkpoint
                          else model_checkpoint)

    if optimizer is not None:
        device = next(model.parameters()).device  # hack to get the device of the model
        optimizer_checkpoint_file = os.path.join(checkpoint_dir_path, "checkpoint.optimizer.pt")
        optimizer_checkpoint = torch.load(optimizer_checkpoint_file, map_location=device)
        optimizer.load_state_dict(optimizer_checkpoint["optimizer_state_dict"])
        del optimizer_checkpoint

    del model_checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_checkpoint(model: Module, optimizer: Optional[Optimizer], checkpoint_dir_path: str, checkpoint_name: str,
                    load_lazy: bool = False):
    _load_checkpoint(model, optimizer, os.path.join(checkpoint_dir_path, checkpoint_name), load_lazy)
