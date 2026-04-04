#!/usr/bin/env python3
"""Story 11.4.6: Evaluate BPE tokenizer against final (default 56-char) toke syntax.

Checks whether key toke syntax patterns are encoded as single BPE tokens,
computes per-line token counts on corpus samples, and writes a JSON report.

This is a READ-ONLY evaluation script -- it does not modify the tokenizer.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

try:
    import sentencepiece as spm
except ImportError:
    print("ERROR: sentencepiece not installed", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Key toke syntax patterns to check
# ---------------------------------------------------------------------------

# These are the patterns that SHOULD be single tokens in a well-trained
# toke BPE tokenizer for the default 56-char syntax.
KEY_PATTERNS = {
    "type_sigils": ["$i64", "$str", "$bool", "$f64", "$u64", "$i32", "$u32", "$mytype"],
    "array_map_open": ["@("],
    "declaration_keywords": ["m=", "f=", "i=", "t="],
    "operators": ["&&", "||", "!=", ">=", "<="],
    "indexing": [".get(", ".len"],
    "old_syntax_keywords": ["M=", "F=", "T=", "I="],
    "old_syntax_types": ["i64", "f64", "Str", "bool", "u64", ":i64", ":f64", ":Str"],
}


def check_pattern_tokenization(sp: spm.SentencePieceProcessor) -> dict:
    """Tokenize each key pattern and report whether it's a single token."""
    results = {}
    for category, patterns in KEY_PATTERNS.items():
        cat_results = []
        for pattern in patterns:
            pieces = sp.encode(pattern, out_type=str)
            ids = sp.encode(pattern, out_type=int)
            cat_results.append({
                "pattern": pattern,
                "pieces": pieces,
                "ids": ids,
                "num_pieces": len(pieces),
                "is_single_token": len(pieces) == 1,
            })
        results[category] = cat_results
    return results


def check_vocab_for_patterns(sp: spm.SentencePieceProcessor) -> dict:
    """Search the full vocabulary for toke-specific tokens."""
    vocab_size = sp.GetPieceSize()
    toke_relevant = {}

    # Patterns to search for in the vocabulary
    search_terms = ["$", "@(", "m=", "f=", "i=", "t=", "M=", "F=", "T=", "I=",
                    "i64", "f64", "u64", "bool", "Str", "str", ".get", ".len",
                    "&&", "||", "mut", "lp(", "el{", "br"]

    for vid in range(vocab_size):
        piece = sp.IdToPiece(vid)
        for term in search_terms:
            if term in piece and vid >= 260:  # skip byte fallback region
                if term not in toke_relevant:
                    toke_relevant[term] = []
                toke_relevant[term].append({"id": vid, "piece": piece})

    return toke_relevant


def load_corpus_samples(corpus_dir: Path, max_samples: int = 200) -> list[str]:
    """Load toke source from corpus JSONL/JSON files."""
    sources = []
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
                    if len(sources) >= max_samples:
                        return sources
        except OSError:
            continue

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
        if len(sources) >= max_samples:
            return sources

    return sources


def analyze_corpus_tokenization(
    sp: spm.SentencePieceProcessor, sources: list[str]
) -> dict:
    """Tokenize corpus samples and compute statistics."""
    token_counts = []
    chars_per_token_list = []
    lines_data = []

    for src in sources:
        ids = sp.encode(src, out_type=int)
        pieces = sp.encode(src, out_type=str)
        n_tokens = len(ids)
        n_chars = len(src)
        token_counts.append(n_tokens)
        if n_tokens > 0:
            chars_per_token_list.append(n_chars / n_tokens)

        # Count how many lines
        n_lines = src.count("\n") + 1
        lines_data.append({
            "n_tokens": n_tokens,
            "n_chars": n_chars,
            "n_lines": n_lines,
            "tokens_per_line": n_tokens / max(n_lines, 1),
        })

    sorted_counts = sorted(token_counts)
    p95_idx = min(int(len(sorted_counts) * 0.95), len(sorted_counts) - 1)

    return {
        "num_samples": len(sources),
        "token_count_mean": round(statistics.mean(token_counts), 1) if token_counts else 0,
        "token_count_median": round(statistics.median(token_counts), 1) if token_counts else 0,
        "token_count_p95": sorted_counts[p95_idx] if sorted_counts else 0,
        "chars_per_token_mean": round(statistics.mean(chars_per_token_list), 3) if chars_per_token_list else 0,
        "tokens_per_line_mean": round(
            statistics.mean(d["tokens_per_line"] for d in lines_data), 1
        ) if lines_data else 0,
    }


def check_new_syntax_in_training_data(train_path: Path) -> dict:
    """Check whether the training data contains new syntax patterns."""
    if not train_path.exists():
        return {"error": f"Training data not found: {train_path}"}

    text = train_path.read_text(encoding="utf-8")
    total_chars = len(text)

    checks = {
        "$": text.count("$"),
        "@(": text.count("@("),
        "m=": text.count("m="),
        "f=": text.count("f="),
        "M=": text.count("M="),
        "F=": text.count("F="),
        "T=": text.count("T="),
        "I=": text.count("I="),
    }

    return {
        "total_chars": total_chars,
        "pattern_counts": checks,
        "uses_new_syntax": checks["$"] > 0 or checks["@("] > 0,
        "uses_old_syntax": checks["M="] > 0 or checks["F="] > 0,
    }


def main() -> int:
    base = Path("/Users/matthew.watt/tk/toke-tokenizer")
    model_path = base / "models" / "toke.model"
    train_path = base / "data" / "train.txt"
    corpus_dir = Path("/Users/matthew.watt/tk/toke-corpus/corpus")
    output_path = base / "docs" / "eval_syntax_tokens_11_4_6.json"

    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 1

    print(f"Loading model from {model_path} ...")
    sp = spm.SentencePieceProcessor()
    sp.Load(str(model_path))
    vocab_size = sp.GetPieceSize()
    print(f"  Vocab size: {vocab_size}")

    # 1. Check training data syntax
    print("\n--- Training Data Syntax Check ---")
    training_check = check_new_syntax_in_training_data(train_path)
    if "error" not in training_check:
        print(f"  Total chars: {training_check['total_chars']}")
        for pat, count in training_check["pattern_counts"].items():
            print(f"  '{pat}' occurrences: {count}")
        print(f"  Uses NEW syntax ($, @()): {training_check['uses_new_syntax']}")
        print(f"  Uses OLD syntax (M=, F=): {training_check['uses_old_syntax']}")

    # 2. Tokenize key patterns
    print("\n--- Key Pattern Tokenization ---")
    pattern_results = check_pattern_tokenization(sp)
    for category, results in pattern_results.items():
        print(f"\n  {category}:")
        for r in results:
            status = "SINGLE" if r["is_single_token"] else f"{r['num_pieces']} pieces"
            print(f"    {r['pattern']:15s} -> {str(r['pieces']):40s} [{status}]")

    # 3. Search vocab for toke-relevant tokens
    print("\n--- Vocab Search ---")
    vocab_search = check_vocab_for_patterns(sp)
    for term, entries in sorted(vocab_search.items()):
        print(f"  '{term}' found in {len(entries)} vocab entries:")
        for e in entries[:5]:
            print(f"    id={e['id']:5d}: {repr(e['piece'])}")
        if len(entries) > 5:
            print(f"    ... and {len(entries) - 5} more")

    # 4. Corpus tokenization stats
    corpus_stats = None
    if corpus_dir.is_dir():
        print("\n--- Corpus Tokenization ---")
        sources = load_corpus_samples(corpus_dir, max_samples=200)
        if sources:
            corpus_stats = analyze_corpus_tokenization(sp, sources)
            print(f"  Samples: {corpus_stats['num_samples']}")
            print(f"  Tokens/sample mean: {corpus_stats['token_count_mean']}")
            print(f"  Tokens/sample median: {corpus_stats['token_count_median']}")
            print(f"  Chars/token mean: {corpus_stats['chars_per_token_mean']}")
            print(f"  Tokens/line mean: {corpus_stats['tokens_per_line_mean']}")
        else:
            print("  No corpus sources found")
    else:
        print(f"\n  Corpus dir not found: {corpus_dir}")

    # 5. Summary verdict
    new_syntax_single = 0
    new_syntax_total = 0
    for cat in ["type_sigils", "array_map_open", "declaration_keywords"]:
        for r in pattern_results[cat]:
            new_syntax_total += 1
            if r["is_single_token"]:
                new_syntax_single += 1

    old_syntax_single = 0
    old_syntax_total = 0
    for cat in ["old_syntax_keywords", "old_syntax_types"]:
        for r in pattern_results[cat]:
            old_syntax_total += 1
            if r["is_single_token"]:
                old_syntax_single += 1

    verdict = {
        "new_syntax_single_token_rate": f"{new_syntax_single}/{new_syntax_total}",
        "old_syntax_single_token_rate": f"{old_syntax_single}/{old_syntax_total}",
        "needs_retraining": new_syntax_single < new_syntax_total * 0.5,
        "reason": "",
    }

    if training_check.get("uses_new_syntax") is False:
        verdict["reason"] = (
            "Tokenizer was trained on OLD syntax (M=, F=, T=, bare type names). "
            "Training data contains zero instances of $ type sigils or @( array syntax. "
            "Retraining on corpus with final syntax is required."
        )
    elif new_syntax_single < new_syntax_total * 0.5:
        verdict["reason"] = (
            "Less than half of new syntax patterns are single tokens. "
            "Retraining recommended."
        )
    else:
        verdict["reason"] = "New syntax patterns are well-represented as single tokens."

    print(f"\n{'=' * 60}")
    print("VERDICT")
    print(f"{'=' * 60}")
    print(f"  New syntax single-token rate: {verdict['new_syntax_single_token_rate']}")
    print(f"  Old syntax single-token rate: {verdict['old_syntax_single_token_rate']}")
    print(f"  Needs retraining: {verdict['needs_retraining']}")
    print(f"  Reason: {verdict['reason']}")

    # Build full report
    report = {
        "story": "11.4.6",
        "title": "BPE tokenizer validation with final syntax",
        "vocab_size": vocab_size,
        "training_data_check": training_check,
        "pattern_tokenization": {
            cat: [
                {
                    "pattern": r["pattern"],
                    "pieces": r["pieces"],
                    "num_pieces": r["num_pieces"],
                    "is_single_token": r["is_single_token"],
                }
                for r in results
            ]
            for cat, results in pattern_results.items()
        },
        "vocab_search_summary": {
            term: len(entries) for term, entries in vocab_search.items()
        },
        "corpus_stats": corpus_stats,
        "verdict": verdict,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
