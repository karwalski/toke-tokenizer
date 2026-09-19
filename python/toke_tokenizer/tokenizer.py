"""BPE tokenizer for the toke programming language."""

from __future__ import annotations

import json
import re
from pathlib import Path

_DATA_PATH = Path(__file__).parent / "data" / "tokenizer_v03.json"

# Module-level cache
_tokenizer: TokeTokenizer | None = None


def _get_tokenizer() -> TokeTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = TokeTokenizer.from_file(_DATA_PATH)
    return _tokenizer


class TokeTokenizer:
    """BPE tokenizer loaded from a HuggingFace tokenizers JSON file."""

    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[tuple[str, str]],
        added_tokens: dict[str, int],
    ) -> None:
        self.vocab = vocab
        self.id_to_token: dict[int, str] = {v: k for k, v in vocab.items()}
        # Also include added tokens in the reverse map
        for token_str, token_id in added_tokens.items():
            self.id_to_token[token_id] = token_str
        self.added_tokens = added_tokens
        self.merges = merges
        # Build merge priority: lower index = higher priority
        self.merge_rank: dict[tuple[str, str], int] = {
            pair: i for i, pair in enumerate(merges)
        }

    @classmethod
    def from_file(cls, path: Path | str) -> TokeTokenizer:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        model = data["model"]
        vocab: dict[str, int] = model["vocab"]
        merges: list[tuple[str, str]] = [
            (parts[0], parts[1]) for parts in model["merges"]
        ]
        added_tokens: dict[str, int] = {
            t["content"]: t["id"] for t in data.get("added_tokens", [])
        }
        return cls(vocab=vocab, merges=merges, added_tokens=added_tokens)

    def _pre_tokenize(self, text: str) -> list[str]:
        """Split text on newlines (isolated), matching the tokenizer's pre-tokenizer."""
        # Split on \n keeping the \n as separate items
        parts = text.split("\n")
        result: list[str] = []
        for i, part in enumerate(parts):
            if part:
                result.append(part)
            if i < len(parts) - 1:
                result.append("\n")
        return result

    def _bpe(self, word: list[str]) -> list[str]:
        """Apply BPE merges to a list of characters/subwords."""
        if len(word) <= 1:
            return word

        while True:
            # Find the pair with the lowest merge rank
            best_pair: tuple[str, str] | None = None
            best_rank = len(self.merges)

            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = self.merge_rank.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            # Merge all occurrences of best_pair
            merged = best_pair[0] + best_pair[1]
            new_word: list[str] = []
            i = 0
            while i < len(word):
                if (
                    i < len(word) - 1
                    and word[i] == best_pair[0]
                    and word[i + 1] == best_pair[1]
                ):
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word

            if len(word) == 1:
                break

        return word

    def encode(self, text: str) -> list[int]:
        """Encode text into a list of token IDs."""
        if not text:
            return []

        tokens: list[int] = []
        chunks = self._pre_tokenize(text)

        for chunk in chunks:
            # Handle newline as a special token
            if chunk == "\n":
                # The vocab has \n at id 3, use it directly
                if "\n" in self.vocab:
                    tokens.append(self.vocab["\n"])
                else:
                    tokens.append(self.added_tokens.get("<newline>", 2))
                continue

            # Split chunk into individual characters as initial BPE word
            chars = list(chunk)
            # Apply BPE
            subwords = self._bpe(chars)
            # Convert to IDs
            for sw in subwords:
                if sw in self.vocab:
                    tokens.append(self.vocab[sw])
                else:
                    # Fallback: encode character by character
                    for ch in sw:
                        if ch in self.vocab:
                            tokens.append(self.vocab[ch])
                        # else: skip unknown characters

        return tokens

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back to text."""
        parts: list[str] = []
        for token_id in ids:
            token = self.id_to_token.get(token_id)
            if token is not None:
                # Skip special tokens in decoded output (except newline)
                if token == "<|endoftext|>" or token == "<pad>":
                    continue
                if token == "<newline>":
                    parts.append("\n")
                else:
                    parts.append(token)
        return "".join(parts)


def _normalise_strings(text: str) -> str:
    """Replace contents of double-quoted strings with underscore."""
    return re.sub(r'"[^"]*"', '"_"', text)


# Public alias
normalise_strings = _normalise_strings


def encode(text: str) -> list[int]:
    """Tokenize text into a list of token IDs."""
    return _get_tokenizer().encode(text)


def decode(ids: list[int]) -> str:
    """Convert token IDs back to text."""
    return _get_tokenizer().decode(ids)


def count_tokens(text: str, normalise_strings: bool = False) -> int:
    """Count the number of tokens in text.

    Args:
        text: The text to tokenize.
        normalise_strings: If True, replace string literal contents with "_"
            before counting.
    """
    if normalise_strings:
        text = _normalise_strings(text)
    return len(encode(text))
