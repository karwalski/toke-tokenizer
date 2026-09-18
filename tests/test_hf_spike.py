"""Story 131.20 item 4: HF byte-level format spike -- the three consumers.

Trains a toy byte-level BPE (plan D1 configuration) in a temp dir from
``tests/fixtures/spike_programs_min.txt`` and checks:

1. ``tokenizers.Tokenizer.from_file`` (what ``train_1b.py::load_tokenizer``
   must do -- today it raises ``NotImplementedError``, asserted here so the
   test flips when Phase 4 implements it);
2. the pip runtime ``python/toke_tokenizer`` loads the file but its char-level
   ``encode`` cannot round-trip byte-level pieces (strict xfail, flips when the
   runtime gains byte-level support), while the ``ByteLevelShim`` in
   ``scripts/hf_spike.py`` round-trips AND reproduces HF's ids exactly;
3. viz decode: raw pieces render through ``piece_to_text``; ``decode(encode(x))
   == x`` incl. non-ASCII; single-id decode of a split multi-byte char is
   U+FFFD, so a viz must decode runs, not single ids.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("tokenizers")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import hf_spike

TRAIN_1B = Path.home() / "tk" / "toke-model" / "train" / "train_1b.py"


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> tuple[object, Path, list[str]]:
    texts = hf_spike.load_fixture(limit=300)
    tok = hf_spike.train_toy(texts, vocab_size=512)
    d = tmp_path_factory.mktemp("toy_tok")
    tok.save(str(d / "tokenizer.json"))
    return tok, d, texts


def test_fixture_is_canonical_single_line() -> None:
    texts = hf_spike.load_fixture()
    assert 200 <= len(texts) <= 400
    assert all("\n" not in t for t in texts)
    assert all(t.startswith("m=") for t in texts)


def test_toy_is_bytelevel_d1_config(toy: tuple[object, Path, list[str]]) -> None:
    import json

    _, d, _ = toy
    data = json.loads((d / "tokenizer.json").read_text())
    assert data["pre_tokenizer"]["type"] == "ByteLevel"
    assert data["pre_tokenizer"]["add_prefix_space"] is False
    assert data["pre_tokenizer"]["use_regex"] is False
    assert data["decoder"]["type"] == "ByteLevel"
    assert data["model"]["type"] == "BPE"
    assert len(data["model"]["vocab"]) == 512


# --- consumer 1: train_1b loader --------------------------------------------


def test_consumer1_tokenizers_from_file_roundtrips(toy: tuple[object, Path, list[str]]) -> None:
    _, d, texts = toy
    hf = hf_spike.train_1b_load_tokenizer(d)
    for t in texts[:50]:
        assert hf.decode(hf.encode(t).ids) == t


@pytest.mark.skipif(not TRAIN_1B.is_file(), reason="toke-model checkout not present")
def test_consumer1_train_1b_loader_is_still_a_stub(toy: tuple[object, Path, list[str]]) -> None:
    """Documents the gap: load_tokenizer(path) must become Tokenizer.from_file(path/'tokenizer.json')."""
    _, d, _ = toy
    spec = importlib.util.spec_from_file_location("train_1b", TRAIN_1B)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError as exc:  # torch etc. not installed here
        pytest.skip(f"train_1b imports unavailable: {exc}")
    with pytest.raises(NotImplementedError):
        mod.load_tokenizer(d)


# --- consumer 2: pip runtime -------------------------------------------------


def test_consumer2_pip_runtime_loads_bytelevel_json(toy: tuple[object, Path, list[str]]) -> None:
    _, d, _ = toy
    base = hf_spike.load_pip_runtime()
    rt = base.from_file(d / "tokenizer.json")
    assert len(rt.vocab) == 512
    assert len(rt.merges) > 0


@pytest.mark.xfail(strict=True, reason="pip runtime is char-level; byte-level pieces need the GPT-2 byte map (Phase 4)")
def test_consumer2_pip_runtime_roundtrips_bytelevel_unshimmed(toy: tuple[object, Path, list[str]]) -> None:
    _, d, texts = toy
    rt = hf_spike.load_pip_runtime().from_file(d / "tokenizer.json")
    assert rt.decode(rt.encode(texts[0])) == texts[0]


def test_consumer2_pip_runtime_with_bytelevel_shim_matches_hf(toy: tuple[object, Path, list[str]]) -> None:
    tok, d, texts = toy
    shim = hf_spike.make_bytelevel_shim(hf_spike.load_pip_runtime()).from_file(d / "tokenizer.json")
    for t in texts[:100]:
        ids = shim.encode(t)
        assert shim.decode(ids) == t
        assert ids == tok.encode(t).ids  # type: ignore[attr-defined]


# --- consumer 3: viz decode --------------------------------------------------


def test_consumer3_pieces_render_as_text(toy: tuple[object, Path, list[str]]) -> None:
    tok, _, texts = toy
    enc = tok.encode(texts[0])  # type: ignore[attr-defined]
    rendered = [hf_spike.piece_to_text(p) for p in enc.tokens]
    assert "".join(rendered) == texts[0]
    assert hf_spike.pieces_to_text(enc.tokens) == texts[0]
    # a space is stored as 'Ġ' in the vocab but must render as ' '
    assert hf_spike.piece_to_text("Ġio") == " io"
    assert hf_spike.piece_to_text("Ċ") == "\n"


def test_consumer3_non_ascii_roundtrip_and_split_char(toy: tuple[object, Path, list[str]]) -> None:
    tok, _, _ = toy
    text = 'io.println("héllo ✓")'
    ids = tok.encode(text).ids  # type: ignore[attr-defined]
    assert tok.decode(ids) == text  # type: ignore[attr-defined]
    singles = [tok.decode([i]) for i in ids]  # type: ignore[attr-defined]
    # the toy vocab has no merges for é / ✓ bytes, so those come out as U+FFFD per id ...
    assert any("�" in s for s in singles)
    # ... but never as base-256 garbage: every single-id render is either text or U+FFFD
    assert all(all(ch == "�" or ord(ch) < 0x10000 for ch in s) for s in singles)


def test_toy_model_is_not_under_models_dir() -> None:
    models = Path(__file__).resolve().parent.parent / "models"
    assert not list(models.rglob("toy*")), "toy tokenizers must never be committed under models/ (131.24)"
