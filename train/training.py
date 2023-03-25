import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.nn import Module
from torch.optim import AdamW

from models.moduleapi import ISparselyWeightDecayedModule
from train import checkpointing, logging
from train.checkpointing import CheckpointInfo


@dataclass
class TrainingConfig:
    # Dataset iterators
    train_dataset_iterator: iter
    val_dataset_iterator: iter

    # Optimizer parameters
    learning_rate: float
    weight_decay: float
    betas: Tuple[float, float]

    # Training loop related parameters
    batch_size: int
    n_mini_steps: int
    evaluation_period: int
    num_evaluation_steps: int

    # Checkpointing parameters
    checkpoint_dir_path: str

    # Miscellaneous parameters
    device: torch.device
    dtype: torch.dtype


def make_batched_iterator(dataset_iterator: iter,
                          batch_size: int,
                          device: torch.device):
    """
    Takes an iterator over individual examples and returns an iterator over batches of examples.
    If the device is of type "cuda", the yielded batches are pinned to memory and non-blocking
    :param dataset_iterator: an infinite iterator over examples (x, y) where x and y are tensors of shape (seq_len,)
    :param batch_size: the number of examples in each batch
    :param device: the device on which to place the yielded batches on
    :return: an infinite iterator over batches of examples (x, y)
            where x and y are tensors of shape (batch_size, seq_len)
    """
    while True:
        examples_x, examples_y = [], []
        for i in range(batch_size):
            x, y = next(dataset_iterator)
            examples_x.append(x)
            examples_y.append(y)
        x, y = torch.stack(examples_x, dim=0), torch.stack(examples_y, dim=0)
        if device.type == "cuda":
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        yield x, y


class LanguageModelTrainer:
    """
    A class responsible for training language models.
    """

    def __init__(self, model: Module, training_config: TrainingConfig):
        model = model.to(training_config.device)

        try:
            torch.compile(model)
        except RuntimeError as e:
            print("Failed to compile model:", e)
            print("Continuing without compilation...")

        self.model = model
        self.training_config = training_config

        device_type = self.training_config.device.type

        # Configure optimizer
        optimizer_param_args = model.parameters()
        if isinstance(model, ISparselyWeightDecayedModule):
            optim_groups = model.get_weight_decay_groups()
            optimizer_param_args = [
                {"params": optim_groups.weight_decay_params, "weight_decay": training_config.weight_decay},
                {"params": optim_groups.no_weight_decay_params, "weight_decay": 0},
            ]
        self.optimizer = AdamW(
            optimizer_param_args,
            lr=self.training_config.learning_rate,
            betas=self.training_config.betas,
            fused=device_type == "cuda"  # fused kernels are only available on CUDA
        )

        # Load checkpoint if it exists
        checkpoint_info = checkpointing.get_checkpoint_info(self.training_config.checkpoint_dir_path, 'latest')
        if checkpoint_info is not None:
            self.current_step = checkpoint_info.step
            # load state dict of model and optimizer
            checkpointing.load_checkpoint(
                self.model, self.optimizer,
                self.training_config.checkpoint_dir_path, "latest"
            )
        else:
            self.current_step = 0
        self.autocast_ctx = nullcontext() if device_type == "cpu" else \
            torch.amp.autocast(device_type=device_type, dtype=self.training_config.dtype)

    def train(self):
        """
        Performs the main training loop of the model
        """

        # Setup dataset iterators
        train_it = make_batched_iterator(dataset_iterator=self.training_config.train_dataset_iterator,
                                         batch_size=self.training_config.batch_size,
                                         device=self.training_config.device)

        # Training loop
        for x, y in train_it:
            step_start_time = time.time()

            # iterate over mini steps
            total_loss = 0
            for ministep in range(self.training_config.n_mini_steps):
                with self.autocast_ctx:
                    logits = self.model(x)
                    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
                loss.backward()
                total_loss += loss.item()

            # update parameters
            with torch.no_grad():
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            step_end_time = time.time()

            step_ms = (step_end_time - step_start_time) * 1000

            # log data from the current step
            log_data = {
                "loss/train": total_loss / self.training_config.n_mini_steps,
            }
            logging.log_train_step(self.current_step, step_ms, log_data)

            if self.current_step % self.training_config.evaluation_period == 0:
                # evaluate model
                eval_loss = self.perform_evaluation()

                # log evaluation data
                log_data = {
                    "loss/val": eval_loss
                }
                logging.log_eval_step(self.current_step, log_data)

                # save checkpoint
                self.save_checkpoint(self.current_step, eval_loss)

            self.current_step += 1

    @torch.no_grad()
    def perform_evaluation(self) -> float:
        """
        Performs evaluation of the model and returns the evaluation loss
        """
        val_it = make_batched_iterator(dataset_iterator=self.training_config.val_dataset_iterator,
                                       batch_size=self.training_config.batch_size,
                                       device=self.training_config.device)

        total_loss = 0
        for i in range(self.training_config.num_evaluation_steps):
            x, y = next(val_it)
            with self.autocast_ctx:
                logits = self.model(x)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
            total_loss += loss.item()

        return total_loss / self.training_config.num_evaluation_steps

    def save_checkpoint(self, step: int, eval_loss: float):
        """
        Performs checkpoint saving for the model.
        Saves the latest checkpoint and the best checkpoint (based on the validation loss)
        :param step: the current step of training
        :param eval_loss: the loss scored during evaluation. Used to determine if the current checkpoint is the best
        """
        best_info = checkpointing.get_checkpoint_info(self.training_config.checkpoint_dir_path, "best")
        current_info = CheckpointInfo(step=step, val_loss=eval_loss)

        # save latest checkpoint
        checkpointing.save_checkpoint(self.model, self.optimizer,
                                      self.training_config.checkpoint_dir_path, "latest",
                                      current_info)

        if best_info is not None and best_info.val_loss < eval_loss:
            return

        # save best checkpoint
        checkpointing.save_checkpoint(self.model, self.optimizer,
                                      self.training_config.checkpoint_dir_path, "best",
                                      current_info)

        logging.log_save_checkpoint(current_info)
