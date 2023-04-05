import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.nn import Module
from torch.optim import AdamW

from models.moduleapi import ISparselyWeightDecayedModule
from train import checkpointing, logging
from utils import iterator_utils
from train.checkpointing import CheckpointInfo
from configure_pytorch import allow_tf32

allow_tf32()


@dataclass
class TrainingConfig:
    # Dataset iterators
    train_dataset_iterator: iter
    val_dataset_iterator: iter

    # Optimizer parameters
    weight_decay: float
    betas: Tuple[float, float]

    # learning rate scheduler parameters
    min_learning_rate: float
    max_learning_rate: float
    warmup_steps: int

    # Training loop related parameters
    batch_size: int
    max_steps: int
    n_mini_steps: int
    evaluation_period: int
    num_evaluation_steps: int

    # Gradient clipping parameters
    grad_clip: float

    # Checkpointing parameters
    checkpoint_dir_path: str

    # Miscellaneous parameters
    device: torch.device
    dtype: torch.dtype

    """
    Whether to delete loss and dependent tensors between mini steps.
    Empties the cuda cache if device is cuda.
    """
    hyper_save_memory: bool = False


def get_learning_rate(step, n_warmup_steps, n_lr_decay_steps, max_lr, min_lr):
    """
    Calculates the learning rate for the given iteration with cosine decay
    :param step: the current step number
    :param n_warmup_steps: the number of iterations to warm up for
    :param n_lr_decay_steps: the number of iterations to decay for (should be ~= max_steps according to Chinchilla)
    """
    # 1) linear warmup for warmup_iters steps
    if step < n_warmup_steps:
        return max_lr * step / n_warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if step > n_lr_decay_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (step - n_warmup_steps) / (n_lr_decay_steps - n_warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (max_lr - min_lr)


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
            lr=self.training_config.max_learning_rate,
            betas=self.training_config.betas,
            fused=device_type == "cuda"  # fused kernels are only available on CUDA
        )
        self.scalar = torch.cuda.amp.GradScaler() if device_type == "cuda" else None

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
        train_it = iterator_utils.prefetching_iterator(
            iterator_utils.make_batched_iterator(
                dataset_iterator=self.training_config.train_dataset_iterator,
                batch_size=self.training_config.batch_size,
                device=self.training_config.device
            ),
            num_prefetch=10
        )

        # Free as much memory as possible before entering the training loop
        self.optimizer.zero_grad(set_to_none=True)
        if self.training_config.device.type == "cuda":
            torch.cuda.empty_cache()

        # Training loop
        while self.current_step < self.training_config.max_steps:
            x, y = next(train_it)
            step_start_time = time.time()

            # update learning rate
            lr = get_learning_rate(
                step=self.current_step,
                n_warmup_steps=self.training_config.warmup_steps,
                n_lr_decay_steps=self.training_config.max_steps,
                max_lr=self.training_config.max_learning_rate,
                min_lr=self.training_config.min_learning_rate
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # iterate over mini steps
            total_loss = 0
            for ministep in range(self.training_config.n_mini_steps):
                with self.autocast_ctx:
                    logits = self.model(x)
                    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))

                total_loss += loss.item()  # use un-scaled loss for logging

                # scale loss for mixed precision training
                if self.scalar is not None:
                    loss = self.scalar.scale(loss)

                loss.backward()

                # free all memory between mini steps to avoid OOM
                del logits
                del loss

                if self.training_config.hyper_save_memory and self.training_config.device.type == "cuda":
                    torch.cuda.empty_cache()

            # Gradient clipping
            if self.training_config.grad_clip != 0.0:
                if self.scalar is not None:
                    self.scalar.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.grad_clip)

            # update parameters
            with torch.no_grad():
                if self.scalar is not None:
                    self.scalar.step(self.optimizer)
                    self.scalar.update()
                else:
                    self.optimizer.step()

                # don't just zero, but free the memory
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
        val_it = iterator_utils.prefetching_iterator(
            iterator_utils.make_batched_iterator(dataset_iterator=self.training_config.val_dataset_iterator,
                                                 batch_size=self.training_config.batch_size,
                                                 device=self.training_config.device),
            num_prefetch=10
        )

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