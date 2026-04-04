#!/usr/bin/env python3
"""Retrain BPE tokenizer on the default-syntax toke corpus.

Reads toke source from corpus_default.jsonl (default-syntax corpus, 46k+ entries),
trains a SentencePiece BPE model at 8k vocab with user-defined symbols for key
syntax patterns, and evaluates single-token coverage and compression.

Usage:
    python retrain_bpe.py \
        --corpus-jsonl /path/to/toke-corpus/data/corpus_default.jsonl \
        --output-dir /tmp/bpe_out
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

try:
    import sentencepiece as spm
except ImportError:
    print(
        "ERROR: sentencepiece is not installed.\n"
        "Install it with:\n"
        "  pip install sentencepiece\n"
        "or, if using the project venv:\n"
        "  .venv/bin/pip install sentencepiece",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants (aligned with existing train.py / prepare.py)
# ---------------------------------------------------------------------------

CHARACTER_COVERAGE = 1.0
USER_DEFINED_SYMBOLS = [
    "$i64", "$f64", "$str", "$bool", "$u64", "$byte",
    "@(",
    "m=", "f=", "t=", "i=",
    ".get(",
    "&&", "||",
]
PAD_ID = -1
UNK_ID = 0
BOS_ID = 1
EOS_ID = 2

# The 13 key patterns that must be single tokens after retraining.
EXPECTED_SINGLE_TOKENS = [
    "$i64", "$f64", "$str", "$bool", "$u64", "$byte",
    "@(",
    "m=", "f=", "t=", "i=",
    ".get(",
    "&&", "||",
]

VOCAB_SIZE = 8000


# ---------------------------------------------------------------------------
# Source collection
# ---------------------------------------------------------------------------

def collect_jsonl_sources(jsonl_path: Path) -> list[str]:
    """Extract tk_source from a JSONL corpus file."""
    sources: list[str] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            source = entry.get("tk_source") or entry.get("toke_source") or entry.get("source")
            if source and isinstance(source, str) and source.strip():
                sources.append(source.strip())
    return sources


def deduplicate(sources: list[str]) -> list[str]:
    """Remove exact duplicates while preserving order."""
    seen: set[str] = set()
    unique: list[str] = []
    for src in sources:
        if src not in seen:
            seen.add(src)
            unique.append(src)
    return unique


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def write_training_text(sources: list[str], path: Path) -> None:
    """Write sources to a text file suitable for SentencePiece training."""
    with open(path, "w", encoding="utf-8") as f:
        for i, src in enumerate(sources):
            if i > 0:
                f.write("\n\n")
            f.write(src.rstrip("\n"))
        f.write("\n")


def train_bpe(input_path: Path, output_dir: Path, vocab_size: int, prefix: str) -> Path:
    """Train a SentencePiece BPE model and return the .model path."""
    model_prefix = str(output_dir / prefix)
    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=model_prefix,
        model_type="bpe",
        vocab_size=vocab_size,
        character_coverage=CHARACTER_COVERAGE,
        user_defined_symbols=",".join(USER_DEFINED_SYMBOLS),
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        normalization_rule_name="identity",
        add_dummy_prefix=False,
        remove_extra_whitespaces=False,
        byte_fallback=True,
    )
    return Path(f"{model_prefix}.model")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def load_model(model_path: Path) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    sp.Load(str(model_path))
    return sp


def char_to_token_ratio(sp: spm.SentencePieceProcessor, sources: list[str]) -> float:
    """Tokens-per-character ratio (fertility) across all sources.

    A lower value means better compression. Target is <= 1.8.
    """
    total_chars = 0
    total_tokens = 0
    for src in sources:
        tokens = sp.encode(src, out_type=int)
        total_chars += len(src)
        total_tokens += len(tokens)
    if total_chars == 0:
        return 0.0
    return total_tokens / total_chars


def check_single_tokens(sp: spm.SentencePieceProcessor, patterns: list[str]) -> dict[str, bool]:
    """Check whether each pattern encodes to a single token."""
    results: dict[str, bool] = {}
    for pattern in patterns:
        pieces = sp.encode(pattern, out_type=str)
        results[pattern] = len(pieces) == 1
    return results


def vocab_coverage(sp: spm.SentencePieceProcessor, sources: list[str]) -> float:
    """Percentage of source text encoded without <unk> tokens."""
    unk_id = sp.unk_id()
    total_tokens = 0
    unk_tokens = 0
    for src in sources:
        ids = sp.encode(src, out_type=int)
        total_tokens += len(ids)
        unk_tokens += ids.count(unk_id)
    if total_tokens == 0:
        return 100.0
    return 100.0 * (1 - unk_tokens / total_tokens)


def roundtrip_fidelity(
    sp: spm.SentencePieceProcessor, sources: list[str], n_samples: int = 100, seed: int = 42
) -> tuple[int, int, list[str]]:
    """Encode then decode n_samples, return (pass_count, total, list of failures)."""
    rng = random.Random(seed)
    samples = rng.sample(sources, min(n_samples, len(sources)))
    passes = 0
    failures: list[str] = []
    for src in samples:
        ids = sp.encode(src, out_type=int)
        decoded = sp.decode(ids)
        if decoded == src:
            passes += 1
        else:
            failures.append(src[:80])
    return passes, len(samples), failures


def tokens_per_line(sp: spm.SentencePieceProcessor, sources: list[str]) -> float:
    """Average tokens per non-empty line across all sources."""
    total_tokens = 0
    total_lines = 0
    for src in sources:
        for line in src.split("\n"):
            if line.strip():
                total_tokens += len(sp.encode(line, out_type=int))
                total_lines += 1
    if total_lines == 0:
        return 0.0
    return total_tokens / total_lines


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(
    vocab_size: int,
    ctr: float,
    single_tok: dict[str, bool],
    coverage: float,
    rt_pass: int,
    rt_total: int,
) -> None:
    """Print a human-readable report for one vocab size."""
    print(f"\n{'=' * 60}")
    print(f"  Vocab size: {vocab_size}")
    print(f"{'=' * 60}")
    target_met = "PASS" if ctr <= 1.8 else "FAIL"
    print(f"  Char-to-token ratio: {ctr:.3f}  (target <= 1.8: {target_met})")
    print(f"  Vocabulary coverage: {coverage:.2f}%")
    print(f"  Round-trip fidelity: {rt_pass}/{rt_total}")
    print(f"  Single-token patterns:")
    for pattern, is_single in single_tok.items():
        status = "YES" if is_single else " no"
        print(f"    {status}  {repr(pattern)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Retrain 8k BPE tokenizer on the default-syntax toke corpus."
    )
    parser.add_argument(
        "--corpus-jsonl",
        type=Path,
        default=Path("/Users/matthew.watt/tk/toke-corpus/data/corpus_default.jsonl"),
        help="Path to the default-syntax corpus JSONL file",
    )
    parser.add_argument(
        "--old-model",
        type=Path,
        default=Path("/Users/matthew.watt/tk/toke-tokenizer/models/toke.model"),
        help="Path to the old tokenizer model for comparison",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/matthew.watt/tk/toke-tokenizer/output"),
        help="Directory for trained models and evaluation results",
    )
    args = parser.parse_args(argv)

    if not args.corpus_jsonl.is_file():
        print(f"ERROR: corpus JSONL not found: {args.corpus_jsonl}", file=sys.stderr)
        return 1

    # ---- Collect sources ----
    print("Loading corpus from JSONL...")
    sources = collect_jsonl_sources(args.corpus_jsonl)
    print(f"  Raw entries: {len(sources)}")
    sources = deduplicate(sources)
    print(f"  After dedup: {len(sources)} unique sources")

    if not sources:
        print("ERROR: no source entries found", file=sys.stderr)
        return 1

    # ---- Prepare training file ----
    models_dir = args.output_dir / "models"
    data_dir = args.output_dir / "data"
    models_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_file = data_dir / "train.txt"
    write_training_text(sources, train_file)
    total_chars = train_file.stat().st_size
    print(f"  Training text: {train_file} ({total_chars:,} bytes)")

    # ---- Train new model ----
    prefix = "toke_default_8k"
    print(f"\nTraining BPE (vocab_size={VOCAB_SIZE}, {len(USER_DEFINED_SYMBOLS)} user-defined symbols)...")
    print(f"  User-defined symbols: {USER_DEFINED_SYMBOLS}")
    model_path = train_bpe(train_file, models_dir, VOCAB_SIZE, prefix)
    print(f"  Model saved: {model_path}")

    # ---- Evaluate new model ----
    sp_new = load_model(model_path)

    ctr = char_to_token_ratio(sp_new, sources)
    single_tok = check_single_tokens(sp_new, EXPECTED_SINGLE_TOKENS)
    coverage = vocab_coverage(sp_new, sources)
    rt_pass, rt_total, rt_failures = roundtrip_fidelity(sp_new, sources, n_samples=200)
    tpl_new = tokens_per_line(sp_new, sources)
    single_count = sum(1 for v in single_tok.values() if v)

    print_report(VOCAB_SIZE, ctr, single_tok, coverage, rt_pass, rt_total)
    print(f"  Avg tokens per line: {tpl_new:.2f}")

    if rt_failures:
        print(f"  Round-trip failures ({len(rt_failures)}):")
        for fail in rt_failures[:5]:
            print(f"    {repr(fail)}")

    # ---- Compare with old model ----
    old_ctr = None
    old_tpl = None
    old_single_tok = None
    if args.old_model.is_file():
        print(f"\n{'=' * 60}")
        print("  COMPARISON WITH OLD MODEL")
        print(f"{'=' * 60}")
        sp_old = load_model(args.old_model)
        old_ctr = char_to_token_ratio(sp_old, sources)
        old_tpl = tokens_per_line(sp_old, sources)
        old_single_tok = check_single_tokens(sp_old, EXPECTED_SINGLE_TOKENS)
        old_single_count = sum(1 for v in old_single_tok.values() if v)

        print(f"  Old model CTR:           {old_ctr:.4f}")
        print(f"  New model CTR:           {ctr:.4f}  (delta: {ctr - old_ctr:+.4f})")
        print(f"  Old tokens/line:         {old_tpl:.2f}")
        print(f"  New tokens/line:         {tpl_new:.2f}  (delta: {tpl_new - old_tpl:+.2f})")
        print(f"  Old single-token:        {old_single_count}/{len(EXPECTED_SINGLE_TOKENS)}")
        print(f"  New single-token:        {single_count}/{len(EXPECTED_SINGLE_TOKENS)}")

        # Show per-pattern comparison
        print(f"\n  Pattern breakdown:")
        for pattern in EXPECTED_SINGLE_TOKENS:
            old_ok = "YES" if old_single_tok[pattern] else " no"
            new_ok = "YES" if single_tok[pattern] else " no"
            changed = " *" if old_single_tok[pattern] != single_tok[pattern] else ""
            print(f"    {old_ok} -> {new_ok}  {repr(pattern)}{changed}")

    # ---- Write evaluation JSON ----
    eval_result = {
        "story": "11.6.2",
        "title": "Retrain BPE on default syntax corpus",
        "vocab_size": VOCAB_SIZE,
        "corpus_entries": len(sources),
        "training_bytes": total_chars,
        "user_defined_symbols": USER_DEFINED_SYMBOLS,
        "new_model": {
            "path": str(model_path),
            "char_to_token_ratio": round(ctr, 4),
            "tokens_per_line": round(tpl_new, 2),
            "single_token_patterns": f"{single_count}/{len(EXPECTED_SINGLE_TOKENS)}",
            "vocab_coverage_pct": round(coverage, 2),
            "roundtrip_fidelity": f"{rt_pass}/{rt_total}",
        },
        "pattern_results": {
            p: {"is_single_token": v, "pieces": sp_new.encode(p, out_type=str)}
            for p, v in single_tok.items()
        },
    }
    if old_ctr is not None:
        eval_result["old_model_comparison"] = {
            "old_ctr": round(old_ctr, 4),
            "new_ctr": round(ctr, 4),
            "ctr_improvement": round(old_ctr - ctr, 4),
            "old_tokens_per_line": round(old_tpl, 2),
            "new_tokens_per_line": round(tpl_new, 2),
            "tpl_improvement": round(old_tpl - tpl_new, 2),
        }

    eval_path = args.output_dir / "eval_retrain_11_6_2.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2)
    print(f"\nEvaluation written to {eval_path}")

    # ---- Copy model to models/ if all 13 patterns pass ----
    if single_count == len(EXPECTED_SINGLE_TOKENS):
        import shutil
        dest = Path("/Users/matthew.watt/tk/toke-tokenizer/models/toke.model")
        dest_backup = dest.with_suffix(".model.old")
        if dest.is_file():
            shutil.copy2(dest, dest_backup)
            print(f"\n  Backed up old model to {dest_backup}")
        shutil.copy2(model_path, dest)
        # Also copy vocab
        vocab_src = model_path.with_suffix(".vocab")
        vocab_dest = Path("/Users/matthew.watt/tk/toke-tokenizer/models/toke.vocab")
        if vocab_src.is_file():
            if vocab_dest.is_file():
                shutil.copy2(vocab_dest, vocab_dest.with_suffix(".vocab.old"))
            shutil.copy2(vocab_src, vocab_dest)
        print(f"  New model installed to {dest}")
        print(f"\n  RESULT: ALL {single_count}/{len(EXPECTED_SINGLE_TOKENS)} patterns are single tokens - PASS")
    else:
        missing = [p for p, v in single_tok.items() if not v]
        print(f"\n  RESULT: {single_count}/{len(EXPECTED_SINGLE_TOKENS)} patterns - INCOMPLETE")
        print(f"  Missing: {missing}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
