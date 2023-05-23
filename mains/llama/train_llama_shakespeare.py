import os
from typing import Tuple

import torch
import numpy as np
from fairscale.nn.model_parallel import initialize_model_parallel

from models.llama import LlamaModel, LlamaConfig
from tokenization.greedy_tokenizer import GreedyTokenizer
from tokenization.sentencepiece_tokenizer import SentencePieceTokenizer
from train.training import TrainingConfig, LanguageModelTrainer
from data.dataset import BinaryTokenDataset


def setup_model_parallel(target_device: torch.device) -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    is_cuda = target_device.type == 'cuda'

    if is_cuda and os.name != 'nt':
        torch.distributed.init_process_group("nccl")
    else:
        torch.distributed.init_process_group("gloo")

    initialize_model_parallel(world_size)

    if is_cuda:
        torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def main():
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    )
    # device = torch.device('cpu')

    tokenizer = SentencePieceTokenizer(model_path='checkpoints/llama/tokenizer.model')
    # tokenizer = GreedyTokenizer.from_json('datasets/shakespeare_char/tokenizer.json')

    local_rank, world_size = setup_model_parallel(target_device=device)

    config = LlamaConfig(
        dim=512,
        multiple_of=256,
        n_heads=8,
        n_layers=8,
        norm_eps=1e-06,
        max_seq_len=1024,
        vocab_size=tokenizer.vocab_size,
        init_weights=True,
    )
    model = LlamaModel(config, device)

    # model = LlamaModel.load('checkpoints/llama/7B',
    #                         tokenizer=tokenizer,
    #                         target_device=device,
    #                         local_rank=local_rank, world_size=world_size,
    #                         fp_16=False)
    # config = model.params

    train_ds = BinaryTokenDataset(
        "datasets/shakespeare_llama/train.bin",
        config.max_seq_len,
        np.dtype(np.uint16)
    )
    val_ds = BinaryTokenDataset(
        "datasets/shakespeare_llama/val.bin",
        config.max_seq_len,
        np.dtype(np.uint16)
    )

    training_config = TrainingConfig(
        train_dataset_iterator=iter(train_ds),
        val_dataset_iterator=iter(val_ds),

        batch_size=1,
        n_mini_steps=1,

        min_learning_rate=1e-6,
        max_learning_rate=1e-4,
        warmup_steps=10,
        max_steps=5000,

        grad_clip=0.0,

        weight_decay=1e-1,
        betas=(0.9, 0.999),

        device=device,
        dtype=torch.float32,

        evaluation_period=100,
        num_evaluation_steps=32,

        checkpoint_dir_path="checkpoints/shakespeare/llama_checkpoints",
    )
    trainer = LanguageModelTrainer(model, training_config)
    trainer.train()


if __name__ == '__main__':
    main()
