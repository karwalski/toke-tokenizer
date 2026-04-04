#!/usr/bin/env python3
"""Analyze overlap between the toke BPE tokenizer and the Qwen base model tokenizer.

Computes vocabulary intersection, Jaccard similarity, identifies novel toke tokens,
and compares per-sample tokenization on corpus programs.  Outputs structured JSON
analysis and a human-readable recommendation.

Usage:
    python tokenizer_alignment.py \
        --toke-model models/toke.model \
        --corpus-dir /path/to/toke-corpus/corpus \
        --output-dir data \
        --sample-count 100

If the ``transformers`` package is not installed, produces partial analysis
with toke BPE statistics only.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

try:
    import sentencepiece as spm
except ImportError:
    print(
        "ERROR: sentencepiece is not installed.\n"
        "Install it with:\n"
        "  pip install sentencepiece",
        file=sys.stderr,
    )
    sys.exit(1)

_HAS_TRANSFORMERS = True
try:
    from transformers import AutoTokenizer  # type: ignore[import-untyped]
except ImportError:
    _HAS_TRANSFORMERS = False

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TOKE_MODEL = "models/toke.model"
DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_CORPUS_DIR = "/Users/matthew.watt/tk/toke-corpus/corpus"
DEFAULT_OUTPUT_DIR = "data"
DEFAULT_SAMPLE_COUNT = 100

# ---------------------------------------------------------------------------
# Corpus collection (reused from retrain_bpe.py)
# ---------------------------------------------------------------------------


def collect_corpus_sources(corpus_dir: Path) -> list[str]:
    """Walk the corpus directory, extracting tk_source from JSON/JSONL files."""
    sources: list[str] = []

    for json_path in sorted(corpus_dir.rglob("*.json")):
        if json_path.name in ("manifest.json", "schema.json"):
            continue
        try:
            with open(json_path, encoding="utf-8") as f:
                entry = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        source = (
            entry.get("tk_source")
            or entry.get("toke_source")
            or entry.get("source")
            or entry.get("code")
        )
        if source and isinstance(source, str) and source.strip():
            sources.append(source.strip())

    for jsonl_path in sorted(corpus_dir.rglob("*.jsonl")):
        try:
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    source = (
                        entry.get("tk_source")
                        or entry.get("toke_source")
                        or entry.get("source")
                        or entry.get("code")
                    )
                    if source and isinstance(source, str) and source.strip():
                        sources.append(source.strip())
        except OSError:
            continue

    return sources


def deduplicate(sources: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for src in sources:
        if src not in seen:
            seen.add(src)
            unique.append(src)
    return unique


# ---------------------------------------------------------------------------
# Tokenizer loading
# ---------------------------------------------------------------------------


def load_toke_tokenizer(model_path: Path) -> spm.SentencePieceProcessor:
    """Load the toke SentencePiece BPE model."""
    if not model_path.exists():
        print(f"ERROR: toke model not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    sp = spm.SentencePieceProcessor()
    sp.Load(str(model_path))
    return sp


def extract_toke_vocab(sp: spm.SentencePieceProcessor) -> set[str]:
    """Extract all vocabulary tokens from a SentencePiece model."""
    vocab: set[str] = set()
    for i in range(sp.GetPieceSize()):
        piece = sp.IdToPiece(i)
        if piece:
            vocab.add(piece)
    return vocab


def load_qwen_tokenizer(model_name: str) -> "AutoTokenizer | None":
    """Load the Qwen tokenizer via transformers, returning None if unavailable."""
    if not _HAS_TRANSFORMERS:
        return None
    try:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as exc:
        print(f"WARNING: could not load Qwen tokenizer: {exc}", file=sys.stderr)
        return None


def extract_qwen_vocab(tokenizer: "AutoTokenizer") -> set[str]:
    """Extract full vocabulary from a HuggingFace tokenizer."""
    vocab_dict = tokenizer.get_vocab()
    return set(vocab_dict.keys())


# ---------------------------------------------------------------------------
# Overlap analysis
# ---------------------------------------------------------------------------


def compute_overlap(toke_vocab: set[str], qwen_vocab: set[str]) -> dict:
    """Compute intersection, union, Jaccard similarity, and novel token sets."""
    intersection = toke_vocab & qwen_vocab
    union = toke_vocab | qwen_vocab
    jaccard = len(intersection) / len(union) if union else 0.0
    novel_in_toke = toke_vocab - qwen_vocab
    overlap_pct = len(intersection) / len(toke_vocab) if toke_vocab else 0.0

    return {
        "toke_vocab_size": len(toke_vocab),
        "qwen_vocab_size": len(qwen_vocab),
        "intersection_size": len(intersection),
        "union_size": len(union),
        "jaccard_similarity": round(jaccard, 6),
        "overlap_pct": round(100.0 * overlap_pct, 2),
        "novel_toke_token_count": len(novel_in_toke),
        "novel_pct": round(100.0 * len(novel_in_toke) / len(toke_vocab), 2) if toke_vocab else 0.0,
        "novel_tokens_sample": sorted(novel_in_toke)[:200],
    }


# ---------------------------------------------------------------------------
# Per-sample comparison
# ---------------------------------------------------------------------------


def compare_tokenization(
    toke_sp: spm.SentencePieceProcessor,
    qwen_tok: "AutoTokenizer",
    sources: list[str],
) -> list[dict]:
    """Tokenize each source with both tokenizers and compare."""
    results: list[dict] = []

    for i, src in enumerate(sources):
        toke_pieces = toke_sp.encode(src, out_type=str)
        qwen_tokens = qwen_tok.tokenize(src)

        toke_set = set(toke_pieces)
        qwen_set = set(qwen_tokens)
        shared = toke_set & qwen_set

        results.append({
            "sample_index": i,
            "source_length_chars": len(src),
            "toke_token_count": len(toke_pieces),
            "qwen_token_count": len(qwen_tokens),
            "ratio_qwen_to_toke": round(len(qwen_tokens) / len(toke_pieces), 3) if toke_pieces else 0.0,
            "shared_unique_tokens": len(shared),
            "toke_only_unique": len(toke_set - qwen_set),
            "qwen_only_unique": len(qwen_set - toke_set),
            "source_preview": src[:120],
        })

    return results


def identify_expansion_patterns(comparisons: list[dict]) -> list[dict]:
    """Find samples where Qwen produces significantly more tokens."""
    expansions = []
    for cmp in comparisons:
        ratio = cmp["ratio_qwen_to_toke"]
        if ratio > 1.5:
            expansions.append({
                "sample_index": cmp["sample_index"],
                "ratio": ratio,
                "toke_tokens": cmp["toke_token_count"],
                "qwen_tokens": cmp["qwen_token_count"],
                "source_preview": cmp["source_preview"],
            })
    return sorted(expansions, key=lambda x: x["ratio"], reverse=True)


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------


def build_recommendation(overlap: dict, comparisons: list[dict] | None) -> dict:
    """Produce a recommendation based on novel-token percentage."""
    novel_pct = overlap["novel_pct"]

    avg_ratio = None
    if comparisons:
        ratios = [c["ratio_qwen_to_toke"] for c in comparisons if c["ratio_qwen_to_toke"] > 0]
        avg_ratio = round(sum(ratios) / len(ratios), 3) if ratios else None

    if novel_pct > 30:
        action = "vocab_extension_prototype"
        summary = (
            f"{novel_pct:.1f}% of toke vocabulary tokens are not in Qwen vocabulary. "
            "Recommend prototyping a vocabulary extension that adds toke-specific "
            "tokens to the base Qwen tokenizer."
        )
        feasibility = "low"
    else:
        action = "use_qwen_tokenizer_directly"
        summary = (
            f"Only {novel_pct:.1f}% of toke vocabulary tokens are novel (not in Qwen). "
            "Using the Qwen tokenizer directly is feasible. The base model already "
            "covers the majority of tokens needed for toke programs."
        )
        feasibility = "high"

    rec = {
        "action": action,
        "novel_token_pct": novel_pct,
        "feasibility_of_qwen_direct": feasibility,
        "summary": summary,
    }
    if avg_ratio is not None:
        rec["avg_qwen_to_toke_token_ratio"] = avg_ratio

    return rec


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_json_report(
    output_dir: Path,
    overlap: dict,
    comparisons: list[dict] | None,
    expansion_patterns: list[dict] | None,
    recommendation: dict,
    partial: bool,
) -> Path:
    """Write the full analysis JSON."""
    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "partial_analysis": partial,
        "overlap_stats": overlap,
        "recommendation": recommendation,
    }
    if comparisons is not None:
        report["per_sample_comparison"] = comparisons
    if expansion_patterns is not None:
        report["expansion_patterns"] = expansion_patterns

    path = output_dir / "tokenizer_alignment.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return path


def write_markdown_report(
    output_dir: Path,
    overlap: dict,
    comparisons: list[dict] | None,
    expansion_patterns: list[dict] | None,
    recommendation: dict,
    partial: bool,
) -> Path:
    """Write the human-readable recommendation markdown."""
    lines: list[str] = []
    lines.append("# Tokenizer Alignment Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    if partial:
        lines.append("")
        lines.append("> **Partial analysis** -- `transformers` package not available.")
        lines.append("> Install with `pip install transformers` for full Qwen comparison.")
    lines.append("")

    lines.append("## Vocabulary Overlap")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Toke vocab size | {overlap['toke_vocab_size']} |")
    if overlap.get("qwen_vocab_size"):
        lines.append(f"| Qwen vocab size | {overlap['qwen_vocab_size']} |")
        lines.append(f"| Intersection | {overlap['intersection_size']} |")
        lines.append(f"| Union | {overlap['union_size']} |")
        lines.append(f"| Jaccard similarity | {overlap['jaccard_similarity']:.4f} |")
        lines.append(f"| Overlap (intersection / toke vocab) | {overlap['overlap_pct']:.1f}% |")
        lines.append(f"| Novel toke tokens | {overlap['novel_toke_token_count']} ({overlap['novel_pct']:.1f}%) |")
    lines.append("")

    if overlap.get("novel_tokens_sample"):
        lines.append("### Novel toke tokens (sample, up to 200)")
        lines.append("")
        lines.append("```")
        for tok in overlap["novel_tokens_sample"][:50]:
            lines.append(f"  {repr(tok)}")
        if len(overlap["novel_tokens_sample"]) > 50:
            lines.append(f"  ... and {len(overlap['novel_tokens_sample']) - 50} more")
        lines.append("```")
        lines.append("")

    if comparisons:
        lines.append("## Per-Sample Tokenization Comparison")
        lines.append("")
        avg_toke = sum(c["toke_token_count"] for c in comparisons) / len(comparisons)
        avg_qwen = sum(c["qwen_token_count"] for c in comparisons) / len(comparisons)
        avg_ratio = sum(c["ratio_qwen_to_toke"] for c in comparisons if c["ratio_qwen_to_toke"] > 0) / max(1, sum(1 for c in comparisons if c["ratio_qwen_to_toke"] > 0))
        lines.append(f"- Samples analyzed: {len(comparisons)}")
        lines.append(f"- Avg toke tokens per sample: {avg_toke:.1f}")
        lines.append(f"- Avg Qwen tokens per sample: {avg_qwen:.1f}")
        lines.append(f"- Avg Qwen/toke token ratio: {avg_ratio:.2f}")
        lines.append("")

    if expansion_patterns:
        lines.append("### Samples where Qwen tokenizer expands significantly (ratio > 1.5x)")
        lines.append("")
        for ep in expansion_patterns[:10]:
            lines.append(f"- Sample {ep['sample_index']}: ratio {ep['ratio']:.2f}x "
                         f"(toke={ep['toke_tokens']}, qwen={ep['qwen_tokens']})")
        lines.append("")

    lines.append("## Recommendation")
    lines.append("")
    lines.append(f"**Action:** `{recommendation['action']}`")
    lines.append("")
    lines.append(recommendation["summary"])
    lines.append("")
    if recommendation.get("avg_qwen_to_toke_token_ratio") is not None:
        lines.append(f"Average Qwen-to-toke token ratio on corpus samples: "
                     f"{recommendation['avg_qwen_to_toke_token_ratio']:.2f}x")
        lines.append("")

    if recommendation["action"] == "vocab_extension_prototype":
        lines.append("### Next Steps")
        lines.append("")
        lines.append("1. Build prototype vocabulary extension adding toke-specific tokens to Qwen tokenizer")
        lines.append("2. Measure downstream LLM perplexity with extended vocab")
        lines.append("3. Compare fine-tuning loss convergence with/without extension")
        lines.append("")
    else:
        lines.append("### Next Steps")
        lines.append("")
        lines.append("1. Confirm Qwen tokenizer covers toke syntax adequately in fine-tuning")
        lines.append("2. Monitor token fertility during training runs")
        lines.append("3. Consider adding only high-frequency novel tokens if fertility is problematic")
        lines.append("")

    path = output_dir / "alignment_recommendation.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze overlap between toke BPE tokenizer and Qwen base model tokenizer."
    )
    parser.add_argument(
        "--toke-model",
        type=Path,
        default=Path(DEFAULT_TOKE_MODEL),
        help=f"Path to toke SentencePiece .model file (default: {DEFAULT_TOKE_MODEL})",
    )
    parser.add_argument(
        "--qwen-model",
        type=str,
        default=DEFAULT_QWEN_MODEL,
        help=f"HuggingFace model ID for Qwen tokenizer (default: {DEFAULT_QWEN_MODEL})",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path(DEFAULT_CORPUS_DIR),
        help="Root of the toke corpus directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f"Directory for output files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help=f"Number of corpus samples for per-sample comparison (default: {DEFAULT_SAMPLE_COUNT})",
    )
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load toke tokenizer ----
    print(f"Loading toke tokenizer from {args.toke_model} ...")
    toke_sp = load_toke_tokenizer(args.toke_model)
    toke_vocab = extract_toke_vocab(toke_sp)
    print(f"  Toke vocab size: {len(toke_vocab)}")

    # ---- Load Qwen tokenizer ----
    qwen_tok = load_qwen_tokenizer(args.qwen_model)
    partial = qwen_tok is None
    if partial:
        print("WARNING: transformers not available -- running partial analysis (toke BPE only)")
        qwen_vocab: set[str] = set()
    else:
        qwen_vocab = extract_qwen_vocab(qwen_tok)
        print(f"  Qwen vocab size: {len(qwen_vocab)}")

    # ---- Overlap analysis ----
    print("\nComputing vocabulary overlap ...")
    overlap = compute_overlap(toke_vocab, qwen_vocab)
    print(f"  Intersection: {overlap['intersection_size']}")
    if not partial:
        print(f"  Jaccard similarity: {overlap['jaccard_similarity']:.4f}")
        print(f"  Overlap (intersection/toke): {overlap['overlap_pct']:.1f}%")
        print(f"  Novel toke tokens: {overlap['novel_toke_token_count']} ({overlap['novel_pct']:.1f}%)")

    # ---- Per-sample comparison ----
    comparisons: list[dict] | None = None
    expansion_patterns: list[dict] | None = None

    if not partial and args.corpus_dir.is_dir():
        print(f"\nCollecting corpus from {args.corpus_dir} ...")
        sources = deduplicate(collect_corpus_sources(args.corpus_dir))
        print(f"  Total unique sources: {len(sources)}")

        n = min(args.sample_count, len(sources))
        if n > 0:
            rng = random.Random(42)
            samples = rng.sample(sources, n)
            print(f"  Comparing tokenization on {n} samples ...")
            comparisons = compare_tokenization(toke_sp, qwen_tok, samples)
            expansion_patterns = identify_expansion_patterns(comparisons)
            if expansion_patterns:
                print(f"  Expansion patterns (Qwen > 1.5x toke): {len(expansion_patterns)}")
        else:
            print("  WARNING: no corpus sources found for per-sample comparison")
    elif not partial:
        print(f"\nWARNING: corpus directory not found: {args.corpus_dir}")

    # ---- Recommendation ----
    recommendation = build_recommendation(overlap, comparisons)
    print(f"\n{'=' * 60}")
    print(f"  RECOMMENDATION: {recommendation['action']}")
    print(f"{'=' * 60}")
    print(f"  {recommendation['summary']}")
    if recommendation.get("avg_qwen_to_toke_token_ratio") is not None:
        print(f"  Avg Qwen/toke token ratio: {recommendation['avg_qwen_to_toke_token_ratio']:.2f}x")

    # ---- Write outputs ----
    json_path = write_json_report(
        args.output_dir, overlap, comparisons, expansion_patterns, recommendation, partial
    )
    print(f"\nJSON report: {json_path}")

    md_path = write_markdown_report(
        args.output_dir, overlap, comparisons, expansion_patterns, recommendation, partial
    )
    print(f"Markdown report: {md_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
