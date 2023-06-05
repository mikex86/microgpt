import numpy as np
import torch
import wandb

from data.dataset import BinaryTokenDataset, HttpBinaryDataset
from models.replit import ReplitLMConfig, ReplitLM
from train import checkpointing
from utils.iterator_utils import make_batched_iterator, prefetching_iterator

BATCH_SIZE = 2

NUM_EVAL_STEPS = 1000


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = ReplitLMConfig(
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

    model = ReplitLM(config)

    try:
        torch.compile(model)
    except RuntimeError as e:
        print("Compilation not available, skipping...")

    checkpointing.load_checkpoint(model, None, 'checkpoints/replit-3b', 'best', load_lazy=True)

    val_dataset = HttpBinaryDataset("http://10.1.1.61:8080/merged.bin", config.max_seq_len, np.uint16)
    val_it = prefetching_iterator(
        make_batched_iterator(iter(val_dataset), batch_size=BATCH_SIZE, device=device),
        num_prefetch=32
    )

    total_loss = 0

    wandb.init(project="replit-perplexity")
    for step in range(NUM_EVAL_STEPS):
        x, y = next(val_it)
        with torch.no_grad():
            y_hat = model(x)

            loss = torch.nn.functional.cross_entropy(
                y_hat.view(-1, config.vocab_size),
                y.view(-1),
                reduction='mean'
            )
            total_loss += loss.item()

            current_loss = total_loss / (step + 1)

            # print(f"Step {step}: {current_loss:.3f}", end='\r')

            wandb.log({"perplexity": current_loss, "tokens": (step + 1) * BATCH_SIZE * config.max_seq_len})


if __name__ == '__main__':
    main()
