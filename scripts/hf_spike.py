#!/usr/bin/env python3
"""HF byte-level tokenizer format spike (story 131.20, plan Phase 0 item 4; D1).

Trains a TOY byte-level BPE with HuggingFace ``tokenizers`` in the D1
configuration (ByteLevel pre-tokenizer, ``add_prefix_space=False``, no GPT-2
regex split, ByteLevel decoder) and exercises the three consumers that must
read the real v0.4 artefact:

1. ``toke-model/train/train_1b.py::load_tokenizer`` -- the intended loader is
   ``tokenizers.Tokenizer.from_file(path / "tokenizer.json")``;
2. the pip runtime ``python/toke_tokenizer/tokenizer.py::TokeTokenizer`` --
   a pure-Python BPE that today assumes CHAR-level pieces; byte-level pieces
   need the GPT-2 byte<->unicode mapping on the way in and out
   (:class:`ByteLevelShim` shows exactly what Phase 4 has to add);
3. a viz decode check -- byte-level pieces (``Ġio``) must be rendered via
   :func:`piece_to_text`, and multi-byte characters can be split across tokens,
   so a viz must decode runs of ids, not single ids.

The toy model is never written under ``models/`` (provenance rule, 131.24);
``python3 scripts/hf_spike.py`` writes it to a temp dir and prints the checks.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "spike_programs_min.txt"
SPECIAL = "<|endoftext|>"


def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) \
        + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs), strict=True))


B2U = bytes_to_unicode()
U2B = {u: b for b, u in B2U.items()}


def text_to_pieces_alphabet(text: str) -> str:
    """Map raw text to the ByteLevel alphabet (what the BPE merges operate on)."""
    return "".join(B2U[b] for b in text.encode("utf-8"))


def piece_to_text(piece: str) -> str:
    """Render one byte-level vocab piece as text (viz mapping).

    Incomplete UTF-8 (a multi-byte char split across pieces) is rendered with
    U+FFFD; a viz should concatenate pieces first (see :func:`pieces_to_text`).
    """
    try:
        return bytes(U2B[c] for c in piece).decode("utf-8", errors="replace")
    except KeyError:
        return piece  # added/special token stored as plain text


def pieces_to_text(pieces: list[str]) -> str:
    return bytes(U2B[c] for p in pieces for c in p if c in U2B).decode("utf-8", errors="replace")


def train_toy(texts: list[str], vocab_size: int = 512) -> Any:
    """Train the D1-configured byte-level BPE on ``texts``."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token=None, byte_fallback=False))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[SPECIAL],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def load_fixture(limit: int | None = None) -> list[str]:
    lines = [ln for ln in FIXTURE.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return lines[:limit] if limit else lines


# ---------------------------------------------------------------------------
# Consumer 2: pure-Python runtime + the shim it is missing
# ---------------------------------------------------------------------------


def load_pip_runtime() -> Any:
    sys.path.insert(0, str(REPO / "python"))
    from toke_tokenizer.tokenizer import TokeTokenizer  # type: ignore[import-not-found]

    return TokeTokenizer


def make_bytelevel_shim(base: Any) -> Any:
    """Subclass the pip runtime so it can read a byte-level HF file.

    Two changes: text -> ByteLevel alphabet before BPE (``encode``), and
    pieces -> bytes -> text after lookup (``decode``).  Everything else in the
    runtime (merge ranks, vocab) already works on the byte-level JSON.
    """

    class ByteLevelShim(base):  # type: ignore[misc]
        def encode(self, text: str) -> list[int]:
            if not text:
                return []
            word = list(text_to_pieces_alphabet(text))
            ids: list[int] = []
            for sw in self._bpe(word):
                ids.append(self.vocab[sw])
            return ids

        def decode(self, ids: list[int]) -> str:
            pieces = [self.id_to_token[i] for i in ids if i in self.id_to_token and self.id_to_token[i] != SPECIAL]
            return pieces_to_text(pieces)

    return ByteLevelShim


# ---------------------------------------------------------------------------
# Consumer 1: train_1b loader
# ---------------------------------------------------------------------------


def train_1b_load_tokenizer(path: Path) -> Any:
    """What ``train_1b.py::load_tokenizer`` needs to do (one line)."""
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path / "tokenizer.json"))


def main() -> int:
    texts = load_fixture()
    tok = train_toy(texts)
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        tok.save(str(out / "tokenizer.json"))
        hf = train_1b_load_tokenizer(out)
        sample = texts[0]
        ids = hf.encode(sample).ids
        print(f"toy vocab={hf.get_vocab_size()} programs={len(texts)}")
        print(f"[1] Tokenizer.from_file roundtrip: {hf.decode(ids) == sample}")
        base = load_pip_runtime()
        rt = base.from_file(out / "tokenizer.json")
        print(f"[2] pip runtime loads file: True; char-level encode roundtrips: {rt.decode(rt.encode(sample)) == sample}")
        shim = make_bytelevel_shim(base).from_file(out / "tokenizer.json")
        print(f"[2] pip runtime + ByteLevelShim roundtrips: {shim.decode(shim.encode(sample)) == sample}; "
              f"ids identical to HF: {shim.encode(sample) == ids}")
        pieces = hf.encode(sample).tokens
        print(f"[3] raw pieces: {pieces[:8]}")
        print(f"[3] viz-rendered: {[piece_to_text(p) for p in pieces[:8]]}")
        print(f"[3] pieces_to_text == input: {pieces_to_text(pieces) == sample}")
        uni = 'io.println("héllo ✓")'
        ids_u = hf.encode(uni).ids
        print(f"[3] non-ASCII decode(encode(x))==x: {hf.decode(ids_u) == uni}; "
              f"single-id renders: {[hf.decode([i]) for i in ids_u][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
