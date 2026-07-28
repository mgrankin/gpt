"""PyTorch 2 implementation of the sparse ruGPT3XL architecture."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .sparse_attention import FixedSparsityConfig, fixed_sparse_attention


@dataclass(frozen=True)
class RuGPT3XLConfig:
    """Architecture settings of the published ruGPT3XL checkpoint."""

    num_layers: int = 24
    vocab_size: int = 50_264
    hidden_size: int = 2_048
    num_attention_heads: int = 16
    max_position_embeddings: int = 2_048
    embedding_dropout: float = 0.1
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    layer_norm_epsilon: float = 1.0e-5
    sparse_mode: str = "alternating"
    sparsity: FixedSparsityConfig = field(
        default_factory=FixedSparsityConfig
    )

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads"
            )
        if self.sparse_mode not in {"all", "alternating", "top_bottom"}:
            raise ValueError(f"Unsupported sparse_mode: {self.sparse_mode}")
        if self.sparsity.num_heads != self.num_attention_heads:
            raise ValueError(
                "The sparsity config and model must have the same head count"
            )
        if self.max_position_embeddings % self.sparsity.local_window:
            raise ValueError(
                "max_position_embeddings must be divisible by the sparse "
                "local window"
            )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def layer_is_sparse(self, layer_index: int) -> bool:
        if self.sparse_mode == "all":
            return True
        if self.sparse_mode == "alternating":
            # Legacy indices 0, 2, ... are sparse.
            return layer_index % 2 == 0
        return layer_index < self.num_layers // 2


class GPT3SelfAttention(nn.Module):
    def __init__(self, config: RuGPT3XLConfig, *, sparse: bool) -> None:
        super().__init__()
        self.config = config
        self.sparse = sparse
        self.query_key_value = nn.Linear(
            config.hidden_size, 3 * config.hidden_size
        )
        self.attention_dropout = config.attention_dropout
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        mixed = self.query_key_value(hidden_states)
        query, key, value = mixed.chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch,
                seq_len,
                self.config.num_attention_heads,
                self.config.head_dim,
            ).permute(0, 2, 1, 3)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)

        if self.sparse:
            context = fixed_sparse_attention(
                query, key, value, self.config.sparsity
            )
        else:
            context = F.scaled_dot_product_attention(
                query,
                key,
                value,
                is_causal=True,
                dropout_p=(
                    self.attention_dropout if self.training else 0.0
                ),
            )

        context = (
            context.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch, seq_len, self.config.hidden_size)
        )
        return self.output_dropout(self.dense(context))


def gelu_openai(tensor: torch.Tensor) -> torch.Tensor:
    """The exact tanh GELU formula used by the legacy Megatron code."""

    return 0.5 * tensor * (
        1.0
        + torch.tanh(
            0.7978845608028654
            * tensor
            * (1.0 + 0.044715 * tensor * tensor)
        )
    )


class GPT3MLP(nn.Module):
    def __init__(self, config: RuGPT3XLConfig) -> None:
        super().__init__()
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size, 4 * config.hidden_size
        )
        self.dense_4h_to_h = nn.Linear(
            4 * config.hidden_size, config.hidden_size
        )
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = gelu_openai(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return self.dropout(hidden_states)


class GPT3TransformerLayer(nn.Module):
    def __init__(self, config: RuGPT3XLConfig, *, sparse: bool) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_epsilon
        )
        self.attention = GPT3SelfAttention(config, sparse=sparse)
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_epsilon
        )
        self.mlp = GPT3MLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attention_input = self.input_layernorm(hidden_states)
        hidden_states = hidden_states + self.attention(attention_input)
        mlp_input = self.post_attention_layernorm(hidden_states)
        return hidden_states + self.mlp(mlp_input)


class GPT3Transformer(nn.Module):
    def __init__(self, config: RuGPT3XLConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            GPT3TransformerLayer(
                config, sparse=config.layer_is_sparse(layer_index)
            )
            for layer_index in range(config.num_layers)
        )
        self.final_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_epsilon
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.final_layernorm(hidden_states)


class RuGPT3XLModel(nn.Module):
    """Checkpoint-compatible ruGPT3XL language model.

    Input sequences are right-padded internally to a 128-token local window.
    Causality guarantees that this cannot affect logits for original tokens.
    """

    def __init__(self, config: RuGPT3XLConfig | None = None) -> None:
        super().__init__()
        self.config = config or RuGPT3XLConfig()
        self.word_embeddings = nn.Embedding(
            self.config.vocab_size, self.config.hidden_size
        )
        self.position_embeddings = nn.Embedding(
            self.config.max_position_embeddings, self.config.hidden_size
        )
        self.embedding_dropout = nn.Dropout(
            self.config.embedding_dropout
        )
        self.transformer = GPT3Transformer(self.config)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        original_length = input_ids.shape[1]
        if original_length < 1:
            raise ValueError("input_ids cannot be empty")
        if original_length > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length {original_length} exceeds the model limit "
                f"of {self.config.max_position_embeddings}"
            )

        window = self.config.sparsity.local_window
        padded_length = ((original_length + window - 1) // window) * window
        padding = padded_length - original_length
        if padding:
            input_ids = F.pad(input_ids, (0, padding), value=0)

        if position_ids is None:
            position_ids = torch.arange(
                padded_length, device=input_ids.device, dtype=torch.long
            ).unsqueeze(0)
        else:
            if position_ids.shape != (input_ids.shape[0], original_length):
                raise ValueError(
                    "position_ids must have the same shape as unpadded input_ids"
                )
            if padding:
                suffix = torch.arange(
                    original_length,
                    padded_length,
                    device=position_ids.device,
                    dtype=position_ids.dtype,
                ).unsqueeze(0)
                suffix = suffix.expand(position_ids.shape[0], -1)
                position_ids = torch.cat((position_ids, suffix), dim=1)

        hidden_states = self.word_embeddings(input_ids)
        hidden_states = hidden_states + self.position_embeddings(position_ids)
        hidden_states = self.embedding_dropout(hidden_states)
        hidden_states = self.transformer(hidden_states)
        hidden_states = hidden_states[:, :original_length]

        if isinstance(logits_to_keep, int):
            if logits_to_keep < 1:
                raise ValueError("logits_to_keep must be positive")
            hidden_states = hidden_states[:, -logits_to_keep:]
        elif isinstance(logits_to_keep, torch.Tensor):
            hidden_states = hidden_states.index_select(1, logits_to_keep)
        elif logits_to_keep is not None:
            raise TypeError("logits_to_keep must be an int, Tensor, or None")

        return F.linear(hidden_states, self.word_embeddings.weight)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        config: RuGPT3XLConfig | None = None,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> "RuGPT3XLModel":
        """Load a legacy consolidated checkpoint without DeepSpeed or Apex."""

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        raw_state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        state = _extract_tensor_state(raw_state)
        normalized = {
            _remove_legacy_prefix(name): tensor
            for name, tensor in state.items()
        }

        with torch.device("meta"):
            model = cls(config)
        incompatible = model.load_state_dict(
            normalized, strict=True, assign=True
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Checkpoint does not match ruGPT3XL: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )

        model = model.to(device=torch.device(device), dtype=dtype)
        return model


def _extract_tensor_state(raw_state: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(raw_state, Mapping) and raw_state and all(
        isinstance(value, torch.Tensor) for value in raw_state.values()
    ):
        return raw_state
    if isinstance(raw_state, Mapping):
        for key in ("module", "model", "state_dict"):
            candidate = raw_state.get(key)
            if isinstance(candidate, Mapping) and candidate and all(
                isinstance(value, torch.Tensor)
                for value in candidate.values()
            ):
                return candidate
    raise TypeError("Checkpoint does not contain a tensor state dictionary")


def _remove_legacy_prefix(name: str) -> str:
    for prefix in ("module.module.", "module."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name
