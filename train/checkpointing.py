import copy
import multiprocessing
import threading
import time
from dataclasses import dataclass
import os
import json
from typing import Optional, Callable

import torch
from torch.nn import Module, Parameter
from torch.optim import Optimizer

from train import logging
from utils import torchhacks


@dataclass
class CheckpointInfo:
    val_loss: float
    step: int
    teacher_eval_loss: Optional[float] = None


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


# launch a separate process to save the checkpoint
class SaveProcess(multiprocessing.Process):

    def __init__(self, save_id, checkpoint_dir_path, checkpoint_info, model_state, optimizer_state):
        super().__init__()
        self.save_id = save_id
        self.checkpoint_dir_path = checkpoint_dir_path
        self.checkpoint_info = checkpoint_info
        self.model_state = model_state
        self.optimizer_state = optimizer_state

    def run(self):
        os.makedirs(self.checkpoint_dir_path, exist_ok=True)
        checkpoint_info_file_path = os.path.join(self.checkpoint_dir_path, "checkpoint_info.json")
        with open(checkpoint_info_file_path, "w") as f:
            json.dump(self.checkpoint_info, f)

        checkpoint_file = os.path.join(self.checkpoint_dir_path, "checkpoint.model.pt")
        with open(checkpoint_file, "wb") as f:
            torch.save({
                "model_state_dict": self.model_state,
            }, checkpoint_file)
            f.flush()
            os.fsync(f)

        checkpoint_file = os.path.join(self.checkpoint_dir_path, "checkpoint.optimizer.pt")
        with open(checkpoint_file, "wb") as f:
            torch.save({
                "optimizer_state_dict": self.optimizer_state,
            }, f)
            f.flush()
            os.fsync(f)


class SaveProcessWatcher(threading.Thread):

    def __init__(self, save_id: any, checkpoint_dir_path: str, process: SaveProcess,
                 on_save_complete: Callable[[], None]):
        super().__init__()
        self.save_id = save_id
        self.checkpoint_dir_path = checkpoint_dir_path
        self.process = process
        self.on_save_complete = on_save_complete

    def run(self):
        start_time = time.time()
        self.process.start()
        self.process.join()
        end_time = time.time()

        self.on_save_complete()
        logging.log_async_save_end(self.save_id, self.checkpoint_dir_path, end_time - start_time)


def _save_checkpoint(model: Module, optimizer: Optimizer, checkpoint_dir_path: str, checkpoint_info: CheckpointInfo,
                     on_save_complete: Callable[[], None]):
    model_state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()  # make a copy of the optimizer state

    device = next(model.parameters()).device  # hack to get the device of the model

    if device.type == 'cuda':
        # copy state dict to cpu via .to(cpu)
        copy_model_state = {}
        for k, v in model_state_dict.items():
            if hasattr(v, 'to') and hasattr(v, 'cpu'):
                copy_model_state[k] = v.to('cpu')
            else:
                copy_model_state[k] = v
        copy_optimizer_state = {}
        for k, v in optimizer_state_dict.items():
            if hasattr(v, 'to') and hasattr(v, 'cpu'):
                copy_optimizer_state[k] = v.to('cpu')
            else:
                copy_optimizer_state[k] = v
    else:
        # deepcopy on cpu
        copy_model_state = copy.deepcopy(model_state_dict)
        copy_optimizer_state = copy.deepcopy(optimizer_state_dict)

    if checkpoint_dir_path in running_save_processes:
        # wait for the previous save process to finish
        if running_save_processes[checkpoint_dir_path].is_alive():
            logging.log_blocking_save(checkpoint_dir_path)
        running_save_processes[checkpoint_dir_path].join()

    save_id = hash(time.time())

    save_process = SaveProcess(save_id, checkpoint_dir_path, checkpoint_info.__dict__, copy_model_state,
                               copy_optimizer_state)
    save_process_watcher = SaveProcessWatcher(save_id, checkpoint_dir_path, save_process, on_save_complete)
    save_process_watcher.start()

    logging.log_async_save_start(save_id, checkpoint_dir_path)

    running_save_processes[checkpoint_dir_path] = save_process


def save_checkpoint_async(model: Module, optimizer: Optimizer, checkpoint_dir_path: str, checkpoint_name: str,
                          checkpoint_info: CheckpointInfo, on_save_complete: Callable[[], None]):
    _save_checkpoint(model, optimizer, os.path.join(checkpoint_dir_path, checkpoint_name), checkpoint_info,
                     on_save_complete)


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
