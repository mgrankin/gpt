"""Production adapter for the original sparse ruGPT3XL checkpoint."""

from __future__ import annotations

from os import getenv
from pathlib import Path

import torch

from front.common import process_seq
from rest.gen import bad_words
from rest.sampling import (
    DEFAULT_TEMPERATURE,
    REPETITION_PENALTY,
    TOP_K,
    TOP_NSIGMA,
    TOP_P,
)
from src.xl_wrapper import RuGPT3XL


MODEL_DIRECTORY = getenv("MODEL", "sparse_xl")
CHECKPOINT_PATH = Path(
    getenv(
        "CHECKPOINT_PATH",
        f"./models/{MODEL_DIRECTORY}/pelevin.model",
    )
)
TOKENIZER_PATH = getenv(
    "TOKENIZER_PATH", "./tokenizer/rugpt3xl.tokenizer"
)
DEVICE = getenv("MODEL_DEVICE", "cuda:0")

if not CHECKPOINT_PATH.is_file():
    raise FileNotFoundError(
        f"Original model checkpoint is missing: {CHECKPOINT_PATH}"
    )

model = RuGPT3XL.from_pretrained(
    TOKENIZER_PATH,
    weights_path=CHECKPOINT_PATH,
    local_files_only=True,
    device=DEVICE,
    dtype=torch.float16,
)
tokenizer = model.tokenizer


def get_sample(
    prompt: str,
    length: int,
    num_samples: int,
    allow_linebreak: bool,
    temperature: float = DEFAULT_TEMPERATURE,
) -> list[str]:
    """Generate continuations using the legacy model's exact sparse layout."""

    max_input = model.model.config.max_position_embeddings - length
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)[-max_input:]
    if not prompt_ids:
        prompt_ids = [model.eos_token_id]

    decoded_prompt = tokenizer.decode(
        prompt_ids, clean_up_tokenization_spaces=False
    )
    outputs = model.generate(
        input_ids=[prompt_ids],
        max_new_tokens=length,
        do_sample=True,
        temperature=temperature,
        top_nsigma=TOP_NSIGMA,
        top_p=TOP_P,
        top_k=TOP_K,
        repetition_penalty=REPETITION_PENALTY,
        bad_words_ids=bad_words(tokenizer, allow_linebreak),
        num_return_sequences=num_samples,
    )

    continuations = []
    for output in outputs:
        if output.startswith(decoded_prompt):
            output = output[len(decoded_prompt) :]
        continuations.append(output)
    return process_seq(continuations)
