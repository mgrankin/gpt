"""Modern implementation of the legacy DeepSpeed fixed sparse attention.

The XL model used DeepSpeed 0.3.7 ``FixedSparsityConfig`` with 16-token
blocks. Current fused attention kernels operate on larger tiles, so passing
the old mask as a dense mask would preserve the result but lose the useful
sparsity. This module instead packs the allowed key/value blocks for every
128-token local window and delegates attention calculation to PyTorch SDPA.

Only the unidirectional pattern used by ruGPT3XL is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FixedSparsityConfig:
    """Parameters of the DeepSpeed 0.3.7 fixed sparsity layout."""

    num_heads: int = 16
    block: int = 16
    different_layout_per_head: bool = True
    num_local_blocks: int = 8
    num_global_blocks: int = 1
    attention: str = "unidirectional"
    horizontal_global_attention: bool = False
    num_different_global_patterns: int = 8

    def __post_init__(self) -> None:
        if self.num_heads < 1:
            raise ValueError("num_heads must be positive")
        if self.block < 1 or self.num_local_blocks < 1:
            raise ValueError("block and num_local_blocks must be positive")
        if self.num_global_blocks < 1:
            raise ValueError("num_global_blocks must be positive")
        if self.num_local_blocks % self.num_global_blocks:
            raise ValueError(
                "num_local_blocks must be divisible by num_global_blocks"
            )
        if self.attention != "unidirectional":
            raise NotImplementedError(
                "The modern XL path implements the checkpoint's "
                "unidirectional sparsity only"
            )
        if self.horizontal_global_attention:
            raise NotImplementedError(
                "horizontal global attention is not part of the XL checkpoint"
            )
        if not self.different_layout_per_head:
            if self.num_different_global_patterns != 1:
                raise ValueError(
                    "Shared head layouts require one global pattern"
                )
        max_patterns = self.num_local_blocks // self.num_global_blocks
        if not 1 <= self.num_different_global_patterns <= max_patterns:
            raise ValueError(
                "num_different_global_patterns must be between 1 and "
                f"{max_patterns}"
            )

    @property
    def local_window(self) -> int:
        return self.block * self.num_local_blocks

    @property
    def global_tokens_per_window(self) -> int:
        return self.block * self.num_global_blocks

    def global_block_offset(self, head: int) -> int:
        """Return the first global block within a local window for ``head``."""

        layout_head = head if self.different_layout_per_head else 0
        pattern = layout_head % self.num_different_global_patterns
        return (
            self.num_local_blocks
            - (1 + pattern) * self.num_global_blocks
        )


def make_legacy_block_layout(
    seq_len: int,
    config: FixedSparsityConfig,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Reproduce ``DeepSpeed==0.3.7`` ``FixedSparsityConfig.make_layout``."""

    if seq_len < 1 or seq_len % config.block:
        raise ValueError(
            f"seq_len ({seq_len}) must be divisible by block ({config.block})"
        )

    num_blocks = seq_len // config.block
    layout = torch.zeros(
        (config.num_heads, num_blocks, num_blocks),
        dtype=torch.bool,
        device=device,
    )
    layout_heads = config.num_heads if config.different_layout_per_head else 1

    for head in range(layout_heads):
        for start in range(0, num_blocks, config.num_local_blocks):
            end = min(start + config.num_local_blocks, num_blocks)
            for row in range(start, end):
                layout[head, row, start : row + 1] = True

        first_global = config.global_block_offset(head)
        complete_end = num_blocks - (num_blocks % config.num_local_blocks)
        for global_start in range(
            first_global, complete_end, config.num_local_blocks
        ):
            layout[
                head,
                global_start:,
                global_start : global_start + config.num_global_blocks,
            ] = True

        if complete_end < num_blocks:
            global_start = min(
                complete_end + first_global,
                num_blocks - config.num_global_blocks,
            )
            layout[
                head,
                global_start:,
                global_start : global_start + config.num_global_blocks,
            ] = True

    if not config.different_layout_per_head:
        layout[1:] = layout[0]
    return layout


def make_legacy_token_mask(
    seq_len: int,
    config: FixedSparsityConfig,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Expand the old block layout and apply its token-level causal mask."""

    block_layout = make_legacy_block_layout(
        seq_len, config, device=device
    )
    token_layout = block_layout.repeat_interleave(
        config.block, dim=-2
    ).repeat_interleave(config.block, dim=-1)
    causal = torch.ones(
        (seq_len, seq_len), dtype=torch.bool, device=device
    ).tril()
    return token_layout & causal


@dataclass(frozen=True)
class _PackedLayout:
    indices: torch.Tensor
    attention_mask: torch.Tensor


@lru_cache(maxsize=32)
def _make_packed_layout_cached(
    seq_len: int,
    config: FixedSparsityConfig,
    device_type: str,
    device_index: int | None,
) -> _PackedLayout:
    device = torch.device(device_type, device_index)
    window = config.local_window
    if seq_len < window or seq_len % window:
        raise ValueError(
            f"seq_len ({seq_len}) must be divisible by the local window "
            f"({window})"
        )

    num_windows = seq_len // window
    global_tokens = config.global_tokens_per_window
    max_keys = window + (num_windows - 1) * global_tokens

    indices = torch.zeros(
        (num_windows, config.num_heads, max_keys),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (num_windows, 1, window, max_keys),
        dtype=torch.bool,
        device=device,
    )
    local_causal = torch.ones(
        (window, window), dtype=torch.bool, device=device
    ).tril()

    for query_window in range(num_windows):
        past_size = query_window * global_tokens
        attention_mask[query_window, 0, :, :past_size] = True
        attention_mask[
            query_window,
            0,
            :,
            past_size : past_size + window,
        ] = local_causal

        for head in range(config.num_heads):
            global_block = config.global_block_offset(head)
            selected: list[int] = []
            for key_window in range(query_window):
                start = (
                    key_window * window + global_block * config.block
                )
                selected.extend(range(start, start + global_tokens))
            selected.extend(
                range(query_window * window, (query_window + 1) * window)
            )
            indices[
                query_window, head, : len(selected)
            ] = torch.tensor(selected, dtype=torch.long, device=device)

    return _PackedLayout(indices=indices, attention_mask=attention_mask)


def _device_cache_key(device: torch.device) -> tuple[str, int | None]:
    if device.type == "cuda" and device.index is None:
        return device.type, torch.cuda.current_device()
    return device.type, device.index


def fixed_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    config: FixedSparsityConfig,
) -> torch.Tensor:
    """Apply the exact XL fixed sparse pattern through PyTorch SDPA.

    Inputs and output use ``[batch, heads, sequence, head_dim]`` layout.
    The sequence must be padded to a full local window. The model wrapper
    performs this harmless right-padding automatically.
    """

    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("fixed sparse attention supports self-attention only")
    if query.ndim != 4:
        raise ValueError("query, key, and value must be rank-4 tensors")

    batch, heads, seq_len, head_dim = query.shape
    if heads != config.num_heads:
        raise ValueError(
            f"Expected {config.num_heads} heads, received {heads}"
        )

    device_type, device_index = _device_cache_key(query.device)
    packed = _make_packed_layout_cached(
        seq_len, config, device_type, device_index
    )
    num_windows = seq_len // config.local_window
    window = config.local_window
    max_keys = packed.indices.shape[-1]

    gather_indices = packed.indices[None, :, :, :, None].expand(
        batch, num_windows, heads, max_keys, head_dim
    )
    expanded_key = key[:, None].expand(
        batch, num_windows, heads, seq_len, head_dim
    )
    expanded_value = value[:, None].expand_as(expanded_key)
    packed_key = torch.gather(
        expanded_key, dim=3, index=gather_indices
    ).reshape(batch * num_windows, heads, max_keys, head_dim)
    packed_value = torch.gather(
        expanded_value, dim=3, index=gather_indices
    ).reshape(batch * num_windows, heads, max_keys, head_dim)

    packed_query = (
        query.reshape(batch, heads, num_windows, window, head_dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch * num_windows, heads, window, head_dim)
    )
    attention_mask = (
        packed.attention_mask[None]
        .expand(batch, num_windows, 1, window, max_keys)
        .reshape(batch * num_windows, 1, window, max_keys)
    )

    output = F.scaled_dot_product_attention(
        packed_query,
        packed_key,
        packed_value,
        attn_mask=attention_mask,
        # The DeepSpeed sparse branch never applied attention dropout.
        dropout_p=0.0,
    )
    return (
        output.reshape(batch, num_windows, heads, window, head_dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch, heads, seq_len, head_dim)
    )
