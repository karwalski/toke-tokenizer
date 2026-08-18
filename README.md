# toke-tokenizer

BPE tokenizer for the [toke programming language](https://github.com/karwalski/toke-spec):
training/evaluation pipeline plus the packaged `toke_tokenizer` Python library.

The current production vocabulary is **v0.3** (16,384 tokens, trained on
default-syntax toke source, ~52% token reduction vs cl100k_base). A v0.4
retrain is planned under Epic 116.9 — see
`toke/docs/architecture/tokenizer-v04-plan.md`. The original design doc
(`docs/tokenizer-design.md`, 32K vocab / 2.5–4x targets) is superseded;
measured results are in the eval reports under `docs/` and `output/`.

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
