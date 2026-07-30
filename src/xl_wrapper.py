"""High-level tokenizer, inference, and generation wrapper for ruGPT3XL."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GPT2Tokenizer

from .modeling_xl import RuGPT3XLModel


DEFAULT_MODEL_ID = "sberbank-ai/rugpt3xl"


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None

    def __getitem__(self, key: str) -> torch.Tensor | None:
        if key == "logits":
            return self.logits
        if key == "loss":
            return self.loss
        raise KeyError(key)


class RuGPT3XL(nn.Module):
    """Compatibility wrapper around the modern PyTorch model.

    Unlike the legacy wrapper, this class does not initialize a distributed
    process group and does not import DeepSpeed or Apex.
    """

    def __init__(
        self,
        model: RuGPT3XLModel,
        tokenizer: GPT2Tokenizer,
        model_path: str,
        *,
        seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.seq_len = seq_len
        vocabulary = tokenizer.get_vocab()
        self.pad_token_id = vocabulary["<pad>"]
        self.eos_token_id = vocabulary["<|endoftext|>"]

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = DEFAULT_MODEL_ID,
        *,
        seq_len: int = 512,
        weights_path: str | Path | None = None,
        deepspeed_config_path: str | Path | None = None,
        local_files_only: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float16,
        **_: object,
    ) -> "RuGPT3XL":
        """Load a tokenizer and local or Hub legacy checkpoint."""

        # Kept for source compatibility with old callers. The exact fixed
        # sparsity configuration now belongs to the architecture.
        del deepspeed_config_path
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.device(device).type == "cpu" and dtype == torch.float16:
            dtype = torch.float32

        tokenizer = GPT2Tokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = "<pad>"

        if weights_path is None:
            from huggingface_hub import hf_hub_download

            weights_path = hf_hub_download(
                repo_id=model_name_or_path,
                filename="mp_rank_00_model_states.pt",
                local_files_only=local_files_only,
            )

        model = RuGPT3XLModel.from_checkpoint(
            weights_path, device=device, dtype=dtype
        )
        model.eval()
        return cls(
            model,
            tokenizer=tokenizer,
            seq_len=seq_len,
            model_path=model_name_or_path,
        )

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, **kwargs: object
    ) -> dict[str, object]:
        kwargs["input_ids"] = input_ids
        return kwargs

    def forward(
        self,
        input_ids: torch.Tensor | Sequence[Sequence[int]] | None = None,
        *,
        text: str | None = None,
        labels: torch.Tensor | Sequence[Sequence[int]] | None = None,
        **_: object,
    ) -> ModelOutput:
        if input_ids is None:
            encoded = self.tokenizer.encode(
                text or "", add_special_tokens=False
            )
            input_ids = [encoded]
        input_tensor = torch.as_tensor(
            input_ids, dtype=torch.long, device=self.device
        )
        logits = self.model(input_tensor)

        loss = None
        if labels is not None:
            label_tensor = torch.as_tensor(
                labels, dtype=torch.long, device=self.device
            )
            if label_tensor.shape != input_tensor.shape:
                raise ValueError("labels and input_ids must have equal shapes")
            loss = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                label_tensor[:, 1:].reshape(-1),
                ignore_index=self.pad_token_id,
            )
        return ModelOutput(logits=logits, loss=loss)

    @torch.inference_mode()
    def generate(
        self,
        text: str | None = None,
        input_ids: torch.Tensor | Sequence[Sequence[int]] | None = None,
        max_length: int | None = None,
        min_length: int | None = None,
        do_sample: bool | None = None,
        early_stopping: bool | None = None,
        num_beams: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        top_nsigma: float | None = None,
        repetition_penalty: float | None = None,
        bad_words_ids: Iterable[Iterable[int]] | None = None,
        bos_token_id: int | None = None,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
        length_penalty: float | None = None,
        no_repeat_ngram_size: int | None = None,
        num_return_sequences: int | None = None,
        decoder_start_token_id: int | None = None,
        use_cache: bool | None = None,
        **model_kwargs: object,
    ) -> list[str]:
        """Generate text with the subset of HF options used by this project."""

        del bos_token_id, length_penalty, decoder_start_token_id, use_cache
        if num_beams not in (None, 1):
            raise NotImplementedError("Beam search is not implemented")

        if input_ids is None:
            encoded = self.tokenizer.encode(
                text or "", add_special_tokens=False
            )
            input_ids = [encoded]
        sequences = torch.as_tensor(
            input_ids, dtype=torch.long, device=self.device
        )
        if sequences.ndim != 2 or sequences.shape[1] == 0:
            raise ValueError("input_ids must be a non-empty rank-2 tensor")

        num_return_sequences = num_return_sequences or 1
        if num_return_sequences > 1:
            sequences = sequences.repeat_interleave(
                num_return_sequences, dim=0
            )

        max_new_tokens = model_kwargs.pop("max_new_tokens", None)
        if model_kwargs:
            unknown = ", ".join(sorted(model_kwargs))
            raise TypeError(f"Unsupported generation options: {unknown}")
        if max_length is None:
            new_tokens = int(max_new_tokens) if max_new_tokens else 20
            max_length = sequences.shape[1] + new_tokens
        max_length = min(
            max_length, self.model.config.max_position_embeddings
        )
        min_length = min_length or 0
        do_sample = bool(do_sample)
        early_stopping = True if early_stopping is None else early_stopping
        temperature = 1.0 if temperature is None else temperature
        top_k = 0 if top_k is None else top_k
        top_p = 1.0 if top_p is None else top_p
        repetition_penalty = (
            1.0 if repetition_penalty is None else repetition_penalty
        )
        no_repeat_ngram_size = no_repeat_ngram_size or 0
        pad_token_id = (
            self.pad_token_id if pad_token_id is None else pad_token_id
        )
        eos_token_id = (
            self.eos_token_id if eos_token_id is None else eos_token_id
        )
        forbidden = [
            tuple(int(token) for token in item)
            for item in (bad_words_ids or ())
            if item
        ]

        finished = torch.zeros(
            sequences.shape[0], dtype=torch.bool, device=self.device
        )
        while sequences.shape[1] < max_length:
            next_logits = self.model(
                sequences, logits_to_keep=1
            )[:, -1].float()
            _apply_repetition_penalty(
                next_logits, sequences, repetition_penalty
            )
            _ban_forbidden_completions(
                next_logits, sequences, forbidden
            )
            if no_repeat_ngram_size > 0:
                _ban_repeated_ngrams(
                    next_logits, sequences, no_repeat_ngram_size
                )
            if sequences.shape[1] < min_length:
                next_logits[:, eos_token_id] = -torch.inf

            if do_sample:
                if temperature <= 0:
                    raise ValueError("temperature must be positive")
                if top_nsigma is not None:
                    _top_nsigma_filter(next_logits, n=top_nsigma)
                next_logits.div_(temperature)
                _top_k_top_p_filter(next_logits, top_k=top_k, top_p=top_p)
                probabilities = torch.softmax(next_logits, dim=-1)
                next_tokens = torch.multinomial(
                    probabilities, num_samples=1
                ).squeeze(1)
            else:
                next_tokens = next_logits.argmax(dim=-1)

            next_tokens = torch.where(
                finished,
                torch.full_like(next_tokens, pad_token_id),
                next_tokens,
            )
            sequences = torch.cat(
                (sequences, next_tokens.unsqueeze(1)), dim=1
            )
            finished |= next_tokens.eq(eos_token_id)
            if early_stopping and bool(finished.all()):
                break

        return [
            self.tokenizer.decode(
                sequence.tolist(), clean_up_tokenization_spaces=False
            )
            for sequence in sequences
        ]


def _apply_repetition_penalty(
    logits: torch.Tensor,
    sequences: torch.Tensor,
    penalty: float,
) -> None:
    if penalty == 1.0:
        return
    if penalty <= 0:
        raise ValueError("repetition_penalty must be positive")
    for row, tokens in enumerate(sequences):
        used = tokens.unique()
        scores = logits[row, used]
        logits[row, used] = torch.where(
            scores < 0, scores * penalty, scores / penalty
        )


def _ban_forbidden_completions(
    logits: torch.Tensor,
    sequences: torch.Tensor,
    forbidden: Sequence[tuple[int, ...]],
) -> None:
    for row, tokens in enumerate(sequences.tolist()):
        for item in forbidden:
            prefix = item[:-1]
            if not prefix or (
                len(tokens) >= len(prefix)
                and tuple(tokens[-len(prefix) :]) == prefix
            ):
                logits[row, item[-1]] = -torch.inf


def _ban_repeated_ngrams(
    logits: torch.Tensor,
    sequences: torch.Tensor,
    ngram_size: int,
) -> None:
    if ngram_size < 1:
        return
    for row, tokens in enumerate(sequences.tolist()):
        prefix_size = ngram_size - 1
        if len(tokens) < prefix_size:
            continue
        prefix = tuple(tokens[-prefix_size:]) if prefix_size else ()
        banned: set[int] = set()
        for start in range(len(tokens) - ngram_size + 1):
            ngram = tuple(tokens[start : start + ngram_size])
            if ngram[:-1] == prefix:
                banned.add(ngram[-1])
        if banned:
            logits[row, list(banned)] = -torch.inf


def _top_k_top_p_filter(
    logits: torch.Tensor, *, top_k: int, top_p: float
) -> None:
    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        threshold = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits.masked_fill_(logits < threshold, -torch.inf)

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True, dim=-1
        )
        cumulative = torch.softmax(
            sorted_logits, dim=-1
        ).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        remove = torch.zeros_like(remove).scatter(
            1, sorted_indices, remove
        )
        logits.masked_fill_(remove, -torch.inf)


def _top_nsigma_filter(logits: torch.Tensor, *, n: float) -> None:
    if not math.isfinite(n) or n <= 0:
        raise ValueError("top_nsigma must be a positive finite number")

    work = logits.float()
    finite = torch.isfinite(work)
    count = finite.sum(dim=-1, keepdim=True)
    safe = torch.where(finite, work, torch.zeros_like(work))
    mean = safe.sum(dim=-1, keepdim=True) / count.clamp_min(1)
    delta = torch.where(finite, work - mean, torch.zeros_like(work))
    denominator = (count - 1).clamp_min(1)
    std = (delta.square().sum(dim=-1, keepdim=True) / denominator).sqrt()
    maximum = work.masked_fill(~finite, -torch.inf).amax(
        dim=-1, keepdim=True
    )
    threshold = maximum - n * std
    logits.masked_fill_(finite & (work < threshold), -torch.inf)
