#!/usr/bin/env python
"""Validate the production sparse ruGPT3XL checkpoint in its Docker image."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer

from src.modeling_xl import RuGPT3XLModel


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/sparse_xl/pelevin.model"),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("dataset/pelevin_valid.txt"),
    )
    parser.add_argument(
        "--tokenizer",
        default="tokenizer/rugpt3xl.tokenizer",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=DTYPES, default="float16")
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--eval-iters",
        type=int,
        default=100,
        help="Legacy evaluation used 100 batches; 0 evaluates one full pass",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--loss-chunk-size", type=int, default=512)
    parser.add_argument("--expected-loss", type=float, default=2.6026)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Report metrics without checking expected-loss tolerance",
    )
    return parser.parse_args()


def tokenize_validation_data(
    path: Path,
    tokenizer: GPT2Tokenizer,
    *,
    seq_length: int,
    seed: int,
) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    complete = len(token_ids) // seq_length
    if complete == 0:
        raise ValueError(
            f"{path} has only {len(token_ids)} tokens, fewer than "
            f"seq_length={seq_length}"
        )
    examples = np.asarray(
        token_ids[: complete * seq_length], dtype=np.int64
    ).reshape(complete, seq_length)
    # Match np.random.seed(1234) + np.random.shuffle from the old loader.
    np.random.RandomState(seed).shuffle(examples)
    return examples


def legacy_batch_loss(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    *,
    pad_token_id: int,
    chunk_size: int,
) -> torch.Tensor:
    """Match the old shifted FP32 vocabulary cross entropy."""

    flat_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
    flat_labels = tokens[:, 1:].reshape(-1)
    # The legacy code shifted labels but sliced the unshifted loss mask.
    flat_mask = tokens[:, :-1].ne(pad_token_id).reshape(-1)
    loss_sum = torch.zeros((), device=logits.device, dtype=torch.float32)
    token_count = torch.zeros(
        (), device=logits.device, dtype=torch.float32
    )
    for start in range(0, flat_labels.numel(), chunk_size):
        end = min(start + chunk_size, flat_labels.numel())
        losses = F.cross_entropy(
            flat_logits[start:end].float(),
            flat_labels[start:end],
            reduction="none",
        )
        mask = flat_mask[start:end]
        loss_sum += losses.masked_select(mask).sum()
        token_count += mask.sum()
    return loss_sum / token_count


def main() -> int:
    args = parse_args()
    if args.seq_length < 1 or args.seq_length > 2048:
        raise ValueError("seq-length must be between 1 and 2048")
    if args.batch_size < 1 or args.loss_chunk_size < 1:
        raise ValueError("batch-size and loss-chunk-size must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    print(f"Loading tokenizer: {args.tokenizer}", flush=True)
    tokenizer = GPT2Tokenizer.from_pretrained(
        args.tokenizer, local_files_only=True
    )
    pad_token_id = tokenizer.get_vocab()["<pad>"]
    examples = tokenize_validation_data(
        args.data,
        tokenizer,
        seq_length=args.seq_length,
        seed=args.seed,
    )
    num_batches = len(examples) // args.batch_size
    if num_batches == 0:
        raise ValueError("batch-size is larger than the validation dataset")
    eval_iters = args.eval_iters or num_batches
    print(
        f"Validation data: {len(examples)} blocks, {num_batches} full "
        f"batches, evaluating {eval_iters} batches",
        flush=True,
    )

    print(
        f"Loading {args.checkpoint} on {device} as {args.dtype}",
        flush=True,
    )
    model = RuGPT3XLModel.from_checkpoint(
        args.checkpoint,
        device=device,
        dtype=DTYPES[args.dtype],
    ).eval()

    started = time.perf_counter()
    total_loss = 0.0
    with torch.inference_mode():
        for iteration in range(eval_iters):
            batch_index = iteration % num_batches
            start = batch_index * args.batch_size
            batch_array = np.array(
                examples[start : start + args.batch_size], copy=True
            )
            tokens = torch.from_numpy(batch_array).to(
                device=device, non_blocking=True
            )
            logits = model(tokens)
            loss = legacy_batch_loss(
                logits,
                tokens,
                pad_token_id=pad_token_id,
                chunk_size=args.loss_chunk_size,
            )
            total_loss += loss.item()
            if (iteration + 1) % 10 == 0 or iteration + 1 == eval_iters:
                print(
                    f"Evaluating {iteration + 1}/{eval_iters}",
                    flush=True,
                )

    mean_loss = total_loss / eval_iters
    perplexity = math.exp(min(20.0, mean_loss))
    elapsed = time.perf_counter() - started
    print("-" * 72)
    print(
        f"validation loss: {mean_loss:.4f} | "
        f"perplexity: {perplexity:.3f} | elapsed: {elapsed:.1f}s"
    )
    print("-" * 72)

    if not args.no_check:
        difference = abs(mean_loss - args.expected_loss)
        if difference > args.tolerance:
            print(
                f"ERROR: loss differs from {args.expected_loss:.4f} by "
                f"{difference:.4f} (tolerance {args.tolerance:.4f})",
                file=sys.stderr,
            )
            return 2
        print(
            f"PASS: loss is within {args.tolerance:.4f} of "
            f"{args.expected_loss:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
