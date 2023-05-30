import json
import multiprocessing
import os

import numpy as np
import torch

from mains.replit.distill import block_prefetcher
from models.replit import ReplitLMConfig, ReplitLM
from train import checkpointing, logging
from train.training import TrainingConfig, LanguageModelTrainer

s3_bucket = 'micro-gpt-datasets-us'
s3_prefix = 'the-stack-replit'

TOKEN_BUDGET = 25 * 10 ** 9


def main():
    logging.set_log_project_name("replit-distill-1b")

    multiprocessing.set_start_method("spawn")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    src_config = ReplitLMConfig(
        d_model=2560,
        n_heads=32,
        n_layers=32,
        mlp_ratio=4,
        max_seq_len=2048,
        vocab_size=32768,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        emb_pdrop=0.0,
        alibi_bias_max=8,
        use_bias=False,
        device=device,
        dtype=torch.bfloat16
    )

    src_model = ReplitLM(src_config)
    checkpointing.load_checkpoint(src_model, None, 'checkpoints/replit-3b', 'best', load_lazy=True)

    dst_config = ReplitLMConfig(
        d_model=1536,
        n_heads=12,
        n_layers=12,
        mlp_ratio=4,
        max_seq_len=2048,
        vocab_size=32768,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        emb_pdrop=0.0,
        alibi_bias_max=8,
        use_bias=False,
        device=device,
        dtype=torch.bfloat16
    )
    dst_model = ReplitLM(dst_config)

    script_dir = os.path.dirname(__file__)
    language_probabilities = json.load(open(os.path.join(script_dir, "language_importance.json"), "r"))

    # normalize probabilities
    total = sum(language_probabilities.values())
    for k in language_probabilities:
        language_probabilities[k] /= total

    # drop 0 probability languages
    language_probabilities = {k: v for k, v in language_probabilities.items() if v > 0}

    train_val_probs = (0.99, 0.01)

    num_blocks_in_flight = 64

    train_it, val_it = block_prefetcher.get_block_iters(f"{s3_bucket}/{s3_prefix}", language_probabilities,
                                                        num_blocks_in_flight,
                                                        src_config.max_seq_len,
                                                        train_val_probs, np.uint16)

    training_config = TrainingConfig(
        train_dataset_iterator=train_it,
        val_dataset_iterator=val_it,

        # teacher model
        src_model=src_model,  # switches into distill mode

        batch_size=7,
        n_mini_steps=5,

        min_learning_rate=6e-6,
        max_learning_rate=6e-4,
        warmup_steps=100,
        max_steps=600000,

        grad_clip=0.01,

        weight_decay=1e-1,
        betas=(0.9, 0.999),

        device=device,
        dtype=torch.float16,

        evaluation_period=50,
        num_evaluation_steps=12,

        checkpoint_dir_path="checkpoints/replit-distill-1b",

        hyper_save_memory=False
    )

    trainer = LanguageModelTrainer(dst_model, training_config)
    trainer.train()


if __name__ == '__main__':
    main()
