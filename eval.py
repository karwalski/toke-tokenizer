#!/usr/bin/env python3
"""Evaluate a trained SentencePiece tokenizer against cl100k_base (tiktoken).

Computes compression ratio, vocabulary utilization, fertility, and
per-program token counts for both the toke tokenizer and baseline.

Metric orientation follows TEMSpec v1.0 §2 (toke-spec/docs/temspec.md):

* ``fertility``        = tokens / character   (§2.4; lower is better)
* ``chars_per_token``  = characters / token   (the inverse; higher is better;
                          this is what the ``>= 1.8`` compression gate reads)
* ``compression_ratio``= sum(toke tokens) / sum(baseline tokens)  (§2.2)
* ``vocab_utilization``= unique tokens used / vocab size          (§2.5)
* ``tokens_per_program``: mean / median / p95 of per-program token counts.
  ``tokens_per_line`` is only meaningful on multi-line source; on ``tkc --min``
  canonical text (one program per line) it degenerates to tokens/program, which
  is why story 131.20 renamed the headline metric.

Usage:
    python eval.py --model models/toke.model --test-data data/valid.txt
    python eval.py --model models/toke.model --test-data data/valid.txt --output eval_report.json
    python eval.py --model models/toke.model --test-data data/valid.txt --dry-run
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any


def load_programs(path: Path) -> list[str]:
    """Load programs from a text file with double-newline separators.

    Matches the write_text() format from prepare.py: programs are
    separated by blank lines (double newlines).
    """
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    # Split on double newlines — each block is one program
    programs = text.split("\n\n")
    # Strip trailing whitespace from each program, drop empty blocks
    return [p.strip() for p in programs if p.strip()]


def tokenize_programs_sp(sp_model: Any, programs: list[str]) -> list[list[int]]:
    """Tokenize each program using a SentencePiece model."""
    return [list(sp_model.encode(p, out_type=int)) for p in programs]


def tokenize_programs_tiktoken(enc: Any, programs: list[str]) -> list[list[int]]:
    """Tokenize each program using a tiktoken encoding."""
    return [list(enc.encode(p)) for p in programs]


def compute_token_stats(token_lists: list[list[int]]) -> dict[str, float]:
    """Compute mean, median, and p95 of token counts per program."""
    if not token_lists:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    counts = [len(toks) for toks in token_lists]
    sorted_counts = sorted(counts)
    p95_idx = int(len(sorted_counts) * 0.95)
    # Clamp index to last element
    p95_idx = min(p95_idx, len(sorted_counts) - 1)
    return {
        "mean": statistics.mean(counts),
        "median": statistics.median(counts),
        "p95": float(sorted_counts[p95_idx]),
    }


def compute_compression_ratio(
    toke_tokens: list[list[int]], baseline_tokens: list[list[int]]
) -> float:
    """Compute compression ratio: total toke tokens / total baseline tokens."""
    toke_total = sum(len(t) for t in toke_tokens)
    baseline_total = sum(len(t) for t in baseline_tokens)
    if baseline_total == 0:
        return 0.0
    return toke_total / baseline_total


def compute_vocab_utilization(
    token_lists: list[list[int]], vocab_size: int
) -> float:
    """Compute vocabulary utilization: unique tokens used / vocab size."""
    if vocab_size == 0:
        return 0.0
    unique = set()
    for toks in token_lists:
        unique.update(toks)
    return len(unique) / vocab_size


# Compression gate (story 131.20): the canonical text must average at least this
# many characters per token.  Expressed in chars/token, NOT tokens/char -- the
# pre-131.20 ``retrain_bpe.char_to_token_ratio`` computed tokens/char (~0.3) and
# compared it against 1.8, so the gate could never fail.
CHARS_PER_TOKEN_GATE_MIN = 1.8


def compute_fertility(programs: list[str], token_lists: list[list[int]]) -> float:
    """Fertility per TEMSpec §2.4: tokens per character, mean over programs.

    Lower is better.  Programs with zero characters are skipped.
    """
    if not programs:
        return 0.0
    ratios = []
    for prog, toks in zip(programs, token_lists):
        char_count = len(prog)
        if char_count > 0:
            ratios.append(len(toks) / char_count)
    if not ratios:
        return 0.0
    return statistics.mean(ratios)


def compute_corpus_fertility(programs: list[str], token_lists: list[list[int]]) -> float:
    """Corpus-level fertility: total tokens / total characters (tokens per char)."""
    total_chars = sum(len(p) for p in programs)
    total_tokens = sum(len(t) for t in token_lists)
    if total_chars == 0:
        return 0.0
    return total_tokens / total_chars


def compute_chars_per_token(programs: list[str], token_lists: list[list[int]]) -> float:
    """Characters per token: total characters / total tokens (higher is better).

    This is the inverse of :func:`compute_corpus_fertility` and the quantity the
    compression gate (:func:`compression_gate`) is defined on.
    """
    total_chars = sum(len(p) for p in programs)
    total_tokens = sum(len(t) for t in token_lists)
    if total_tokens == 0:
        return 0.0
    return total_chars / total_tokens


def compression_gate(
    chars_per_token: float, minimum: float = CHARS_PER_TOKEN_GATE_MIN
) -> bool:
    """Return True when ``chars_per_token`` (chars/token) meets the gate minimum.

    Takes chars/token, never tokens/char: passing a fertility value here is a
    bug (it would sit around 0.3 and always fail, the mirror image of the
    pre-131.20 bug where tokens/char was compared against 1.8 and always passed).
    """
    if chars_per_token <= 0:
        return False
    return chars_per_token >= minimum


def compute_tokens_per_program(token_lists: list[list[int]]) -> dict[str, float]:
    """Mean / median / p95 tokens per program (alias of :func:`compute_token_stats`)."""
    return compute_token_stats(token_lists)


def compute_tokens_per_line(
    programs: list[str], encode: Callable[[str], list[int]]
) -> float:
    """Mean tokens per non-empty source LINE, tokenising each line separately.

    Only meaningful for multi-line (readable) source.  On ``tkc --min`` output
    every program is a single line, so this equals tokens/program -- report
    :func:`compute_tokens_per_program` there instead.
    """
    total_tokens = 0
    total_lines = 0
    for prog in programs:
        for line in prog.split("\n"):
            if line.strip():
                total_tokens += len(encode(line))
                total_lines += 1
    if total_lines == 0:
        return 0.0
    return total_tokens / total_lines


def build_report(
    programs: list[str],
    toke_tokens: list[list[int]],
    baseline_tokens: list[list[int]],
    vocab_size: int,
) -> dict[str, Any]:
    """Build the full evaluation report as a dictionary."""
    return {
        "program_count": len(programs),
        "toke": compute_token_stats(toke_tokens),
        "baseline": compute_token_stats(baseline_tokens),
        "compression_ratio": compute_compression_ratio(toke_tokens, baseline_tokens),
        "vocab_utilization": compute_vocab_utilization(toke_tokens, vocab_size),
        "fertility": compute_fertility(programs, toke_tokens),
        "corpus_fertility": compute_corpus_fertility(programs, toke_tokens),
        "chars_per_token": compute_chars_per_token(programs, toke_tokens),
        "compression_gate_min_chars_per_token": CHARS_PER_TOKEN_GATE_MIN,
        "compression_gate_pass": compression_gate(
            compute_chars_per_token(programs, toke_tokens)
        ),
        "vocab_size": vocab_size,
    }


def format_summary(report: dict[str, Any]) -> str:
    """Format a human-readable summary table from the report."""
    lines = [
        "Tokenizer Evaluation Report",
        "=" * 40,
        f"Programs evaluated:    {report['program_count']}",
        f"Vocabulary size:       {report['vocab_size']}",
        "",
        "Tokens per program         toke     baseline",
        "-" * 40,
        f"  mean                 {report['toke']['mean']:>8.1f}   {report['baseline']['mean']:>8.1f}",
        f"  median               {report['toke']['median']:>8.1f}   {report['baseline']['median']:>8.1f}",
        f"  p95                  {report['toke']['p95']:>8.1f}   {report['baseline']['p95']:>8.1f}",
        "",
        f"Compression ratio:     {report['compression_ratio']:.4f}",
        f"Vocab utilization:     {report['vocab_utilization']:.4f}",
        f"Fertility (tok/char):  {report['fertility']:.4f}",
    ]
    if "chars_per_token" in report:
        gate = "PASS" if report.get("compression_gate_pass") else "FAIL"
        lines.append(
            f"Chars per token:       {report['chars_per_token']:.4f}"
            f"  (gate >= {report['compression_gate_min_chars_per_token']}: {gate})"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a toke SentencePiece tokenizer against cl100k_base."
    )
    parser.add_argument(
        "--model", required=True, type=Path, help="Path to toke .model file"
    )
    parser.add_argument(
        "--test-data", required=True, type=Path, help="Path to test data (valid.txt)"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Path to write JSON report"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="cl100k_base",
        help="Tiktoken encoding name (default: cl100k_base)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print config without running evaluation",
    )
    args = parser.parse_args(argv)

    # Validate inputs
    if not args.model.exists():
        print(f"ERROR: model file not found: {args.model}", file=sys.stderr)
        return 1
    if not args.test_data.exists():
        print(f"ERROR: test data not found: {args.test_data}", file=sys.stderr)
        return 1

    # Load programs
    programs = load_programs(args.test_data)
    if not programs:
        print("ERROR: no programs found in test data", file=sys.stderr)
        return 1

    # Dry-run mode
    if args.dry_run:
        print("Dry-run mode — validating inputs only")
        print(f"  Model:     {args.model}")
        print(f"  Test data: {args.test_data}")
        print(f"  Baseline:  {args.baseline}")
        print(f"  Output:    {args.output or '(stdout only)'}")
        print(f"  Programs:  {len(programs)}")
        return 0

    # Load tokenizers (deferred imports so dry-run works without deps)
    try:
        import sentencepiece as spm
    except ImportError:
        print("ERROR: sentencepiece is not installed", file=sys.stderr)
        return 1
    try:
        import tiktoken  # type: ignore[import-untyped]
    except ImportError:
        print("ERROR: tiktoken is not installed", file=sys.stderr)
        return 1

    # Load toke model
    sp = spm.SentencePieceProcessor()
    sp.Load(str(args.model))
    vocab_size = sp.GetPieceSize()

    # Load baseline
    enc = tiktoken.get_encoding(args.baseline)

    # Tokenize
    toke_tokens = tokenize_programs_sp(sp, programs)
    baseline_tokens = tokenize_programs_tiktoken(enc, programs)

    # Build report
    report = build_report(programs, toke_tokens, baseline_tokens, vocab_size)

    # Output
    summary = format_summary(report)
    print(summary)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
