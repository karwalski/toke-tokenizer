# Contributing to toke-tokenizer

## Rules

- Tokenizer vocabulary files (.model, .vocab) are never committed to git.
- All Python code must pass `ruff check .` and `mypy .` before commit.
- Changes to tokenizer training parameters require a documented rationale
  in docs/tokenizer-design.md.

## Tokenizer artefact provenance

- Every file under `models/` and every `tokenizer*.json` at the repo root is an artefact.
- Each artefact ships with a committed `<name>.provenance.json` beside it
  (or a `provenance.json` in its directory).
- The record names the committed training script, its git blob/commit SHA, the config SHA
  and the corpus manifest SHA that produced the bytes.
- `python scripts/check_provenance.py` (also `make check-provenance`; runs in CI) fails on any
  artefact without a valid record or whose `sha256` no longer matches.
- Artefacts that predate this rule are marked `"legacy": true, "provenance": "unknown (legacy)"`;
  never add new ones that way.

Required keys: `artifact`, `sha256`, `training_script`, `training_script_sha`, `config_sha`,
`corpus_manifest_sha`, `created`, `created_by`. The four training/config/manifest keys may be
`"unknown (legacy)"` only on a `"legacy": true` record. Background: `tokenizer_v03.json` shipped
with no committed training script (see `toke/docs/architecture/tokenizer-v04-plan.md`).

## Testing

    python -m pytest tests/ -v

## Developer Certificate of Origin

Sign your commits: `git commit -s`
