# toke-tokenizer

BPE tokenizer for the [toke programming language](https://github.com/karwalski/toke-spec):
training/evaluation pipeline plus the packaged `toke_tokenizer` Python library.

The current production vocabulary is **v0.3** (16,384 tokens, trained on
default-syntax toke source). A v0.4 retrain is planned under Epic 116.9 — see
`toke/docs/architecture/tokenizer-v04-plan.md`. The original design doc
(`docs/tokenizer-design.md`, 32K vocab / 2.5–4x targets) is superseded;
measured results are in the eval reports under `docs/` and `output/`.

## About toke

> toke: a compiled language designed for LLM code generation, with a small grammar, one
> canonical form and compiler verification.

toke is a compiled programming language designed for LLM code generation. It has 14
keywords, a 55-character set, a backtrack-free grammar with bounded lookahead, and one
canonical form per construct, chosen by measurement in a 46-pattern catalogue and
reproduced by `tkc --min`. That makes generated code cheap to constrain during decoding,
cheap for a compiler to verify afterwards, and compact to emit. Token efficiency is one
measured property of toke, always reported with its tokenizer and its baseline, not the
whole claim.

*The one-liner and the paragraph above are reproduced word for word from the canonical
description,
[`docs/about/canonical.md`](https://github.com/karwalski/toke/blob/main/docs/about/canonical.md).
Every number published about toke comes from
[`docs/metrics-baseline.md`](https://github.com/karwalski/toke/blob/main/docs/metrics-baseline.md)
and nowhere else.*

## Token efficiency — read this before quoting a number

**No toke tokenizer shipped so far beats cl100k_base.** On canonical `tkc --min`
v0.4 text with string bodies masked (N = 2,000 stratified corpus records,
2026-09-18, `docs/baseline_v04_pre131.md`):

| tokenizer | tokens/program | vs cl100k_base |
|---|---:|---:|
| cl100k_base | 121.2 [118.9, 123.6] | 1.000 |
| o200k_base | 122.6 | 1.012 |
| Qwen2.5-Coder | 125.1 | 1.032 |
| **SentencePiece 8k (shipped)** | 139.8 | **1.154 — 15.4% MORE tokens** |
| SentencePiece 32k | 139.6 | 1.152 |
| `tokenizer_v03.json` (16,384) | 66.0 | 0.545 — **lossy, not creditable** |

`tokenizer_v03.json` only appears to win because its `unk_token` is `null`: it
silently drops every backslash (2,606 of them in that sample) and therefore
under-counts. Its row is informational only.

The **"~52% token reduction vs cl100k_base"** that this README and the PyPI
description used to carry is **withdrawn**. As published it meant Toke-16K
encoding toke source versus cl100k_base encoding *the same toke source* —
one text, two tokenizers (TEMSpec §2.2), N = 42 v0.3 benchmark programs — and it
was never a comparison with Python. It is withdrawn because it is superseded by
the v0.4 measurement above *and* because it does not reconcile with its own
published dataset (`toke/docs/reference/token-comparison.md`, the same N = 42
set, gives 61.6%). A "purpose-built tokenizer beats cl100k" claim becomes
supportable only when 116.9 trains and locks a v0.4 vocabulary against the
pinned baseline (cl100k_base = 242,427 tokens on that sample).

**Cross-language numbers.** A toke-trained tokenizer must **never** be run over
Python, C or Java source: it is out of domain there and inflates the baseline
mechanically, measuring its own training bias. Any toke-vs-other-language figure
uses one tokenizer on both sides. Under cl100k_base on both sides toke currently
costs **1.34x [1.22, 1.48]** the tokens of equivalent Python over the 60 Gate-1
tasks (N = 60) — more, not fewer. Canonical wording for any public claim:
`toke/docs/metrics-baseline.md`.

## Layout

| Path | What it is |
|------|------------|
| `prepare.py` / `train.py` / `eval.py` | Corpus prep, SentencePiece BPE training wrapper, evaluation vs cl100k_base |
| `scripts/retrain_bpe.py` | Retrains BPE on the default-syntax corpus (`corpus_default.jsonl`), with single-token coverage + compression eval |
| `scripts/eval_syntax_tokens.py`, `scripts/tokenizer_alignment.py` | Syntax-token coverage eval; vocab overlap analysis vs base-model tokenizers |
| `python/` | Packaged `toke_tokenizer` library (pure-Python encode/decode/count_tokens; pip-installable, own pyproject/README) |
| `tokenizer_v03.json` | v0.3 vocab retrain (May 2026), HF tokenizer format. Kept at repo root pending the v0.4 retrain (116.9) |
| `tests/` | Unit tests for prepare/train/eval |

## Usage

    # Prepare corpus for tokenizer training (JSONL -> train.txt / valid.txt)
    python prepare.py --input /path/to/corpus.jsonl --output-dir tokenizer-data/

    # Train the tokenizer (SentencePiece BPE)
    python train.py --input tokenizer-data/train.txt --output-dir models/ --vocab-size 16384

    # Evaluate against cl100k_base
    python eval.py --model models/toke.model --test-data tokenizer-data/valid.txt --output report.json

    # Retrain on the default-syntax corpus
    python scripts/retrain_bpe.py \
        --corpus-jsonl /path/to/toke-corpus/data/corpus_default.jsonl \
        --output-dir /tmp/bpe_out

Optional dependencies: `pip install .[train]` (sentencepiece) or `.[eval]`
(sentencepiece + tiktoken).

## Data

Large artifacts are gitignored and live locally only: `data/` and `models/`
directories (at any depth, including under `output/`), `*.jsonl`, and
`tokenizer-data/`. Only small JSON eval reports are tracked under `docs/` and
`output/`. Archived snapshots of the legacy corpus/model era are at
`~/tk/archive/toke-tokenizer-legacy-20260819`.

## Related repos

- **toke-corpus** — source of the training text (validated toke programs).
- **toke-mcp**, **toke-website** — consume the trained vocabulary for token
  counting and display.

## Licence

Apache 2.0 (the `python/` package ships under its own MIT licence).
