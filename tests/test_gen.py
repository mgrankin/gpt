import tempfile
import unittest
from pathlib import Path

from rest.gen import TextDataset, get_line_enders, iftoken


class FakeTokenizer:
    vocab = {
        "plain": 1,
        "line\n": 2,
        "\n": 3,
    }

    def encode(self, text, add_special_tokens=True):
        del add_special_tokens
        known = {
            "one": [10],
            "two tokens": [20, 21],
            "three": [30],
        }
        return known.get(text, [ord(character) for character in text])

    def decode(self, token_id):
        return next(
            token for token, current_id in self.vocab.items()
            if current_id == token_id
        )

    def get_vocab(self):
        return self.vocab


class GenHelpersTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def test_iftoken_keeps_only_single_token_values(self):
        self.assertEqual(
            iftoken(self.tokenizer, ["one", "two tokens", "three"]),
            [10, 30],
        )

    def test_get_line_enders_returns_decoded_tokens_with_newline(self):
        self.assertEqual(
            get_line_enders(self.tokenizer),
            ["line\n", "\n"],
        )

    def test_text_dataset_splits_complete_sequences(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            path.write_text("abcdefghi")
            dataset = TextDataset(path, self.tokenizer, seq_length=4)

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0].tolist(), [97, 98, 99, 100])
        self.assertEqual(dataset[1].tolist(), [101, 102, 103, 104])


if __name__ == "__main__":
    unittest.main()
