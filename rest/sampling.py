"""Shared sampling settings and logits processors."""

from __future__ import annotations

import math

import torch


DEFAULT_TEMPERATURE = 1.1
REPETITION_PENALTY = 1.2
TOP_NSIGMA = 1.2
TOP_P = 1.0
TOP_K = 0


class TopNSigmaLogitsProcessor:
    """Keep logits no further than ``n`` standard deviations from the max."""

    def __init__(self, n: float = TOP_NSIGMA) -> None:
        if not math.isfinite(n) or n <= 0:
            raise ValueError("n must be a positive finite number")
        self.n = float(n)

    def __call__(
        self,
        input_ids: object,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        del input_ids
        if scores.ndim not in (1, 2):
            raise ValueError("scores must be a rank-1 or rank-2 tensor")

        rows = scores.unsqueeze(0) if scores.ndim == 1 else scores
        work = rows.float()
        finite = torch.isfinite(work)
        count = finite.sum(dim=-1, keepdim=True)

        # Some backends run custom processors after bad-word processors. Ignore
        # their -inf masks when calculating sigma, while preserving the masks.
        safe = torch.where(finite, work, torch.zeros_like(work))
        mean = safe.sum(dim=-1, keepdim=True) / count.clamp_min(1)
        delta = torch.where(finite, work - mean, torch.zeros_like(work))
        denominator = (count - 1).clamp_min(1)
        std = (delta.square().sum(dim=-1, keepdim=True) / denominator).sqrt()
        maximum = work.masked_fill(~finite, -torch.inf).amax(
            dim=-1, keepdim=True
        )
        threshold = maximum - self.n * std
        remove = finite & (work < threshold)
        return scores.masked_fill(
            remove.squeeze(0) if scores.ndim == 1 else remove,
            -torch.inf,
        )
