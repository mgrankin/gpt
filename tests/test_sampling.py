from __future__ import annotations

import unittest

import torch

from front.common import Prompt
from rest.sampling import (
    DEFAULT_TEMPERATURE,
    REPETITION_PENALTY,
    TOP_K,
    TOP_NSIGMA,
    TOP_P,
    TopNSigmaLogitsProcessor,
)
from src.xl_wrapper import _top_nsigma_filter


class TopNSigmaTests(unittest.TestCase):
    def test_project_sampling_defaults(self) -> None:
        self.assertEqual(TOP_NSIGMA, 1.2)
        self.assertEqual(DEFAULT_TEMPERATURE, 1.1)
        self.assertEqual(REPETITION_PENALTY, 1.2)
        self.assertEqual(TOP_P, 1.0)
        self.assertEqual(TOP_K, 0)
        self.assertEqual(Prompt().temperature, DEFAULT_TEMPERATURE)

    def test_filters_below_max_minus_nsigma(self) -> None:
        logits = torch.tensor([[10.0, 9.0, 8.0, 0.0]])
        expected_std = logits.std(dim=-1, keepdim=True)
        expected = logits < logits.max(dim=-1, keepdim=True).values - expected_std

        result = TopNSigmaLogitsProcessor(1.0)(None, logits)

        self.assertTrue(torch.equal(torch.isneginf(result), expected))

    def test_candidate_set_is_temperature_invariant(self) -> None:
        logits = torch.tensor([[7.0, 5.0, 4.0, 1.0, -3.0]])
        processor = TopNSigmaLogitsProcessor(1.2)

        base = torch.isfinite(processor(None, logits))
        heated = torch.isfinite(processor(None, logits / 3.0))

        self.assertTrue(torch.equal(base, heated))

    def test_preserves_existing_masks_and_ignores_them_in_std(self) -> None:
        logits = torch.tensor([10.0, 9.0, 0.0, -torch.inf])

        result = TopNSigmaLogitsProcessor(1.0)(None, logits)

        self.assertTrue(torch.isfinite(result[0]))
        self.assertTrue(torch.isfinite(result[1]))
        self.assertTrue(torch.isneginf(result[2]))
        self.assertTrue(torch.isneginf(result[3]))

    def test_xl_filter_matches_shared_processor(self) -> None:
        logits = torch.tensor(
            [[8.0, 6.0, 5.0, -2.0], [4.0, 3.5, 0.0, -torch.inf]]
        )
        expected = TopNSigmaLogitsProcessor(1.2)(None, logits)
        actual = logits.clone()

        _top_nsigma_filter(actual, n=1.2)

        self.assertTrue(torch.equal(actual, expected))

    def test_rejects_invalid_n(self) -> None:
        for value in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    TopNSigmaLogitsProcessor(value)


if __name__ == "__main__":
    unittest.main()
