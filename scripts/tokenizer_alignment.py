#!/usr/bin/env python3
"""Vocabulary alignment between the toke tokenizer and the Qwen2.5-Coder tokenizer.

Story 131.20 (plan Phase 0 item 2) rewrite of the 9.8.2 script.  Fixes:

* ``transformers`` is REQUIRED.  The 9.8.2 run silently degraded to a
  "partial analysis" with ``qwen_vocab_size: 0`` and still emitted a
  ``vocab_extension_prototype`` verdict -- that verdict was an artefact.  This
  script now exits 2 with a clear message if ``transformers`` is missing, and
  never writes a report without the Qwen side.
* Piece representations are normalised before any set comparison: SentencePiece
  ``▁`` (word-boundary marker) and ``<0xNN>`` byte-fallback pieces, and
  HuggingFace byte-level pieces (``Ġ`` = space, ``Ċ`` = newline, ...)
  are all mapped to the surface text they stand for.  Without this, ``▁io``
  and ``Ġio`` are counted as different tokens although both mean `` io``.
* Adds an occurrence-weighted coverage: the fraction of toke token
  *occurrences* over the canonical (``tkc --min`` + masked, plan D2/D6) sample
  whose surface text is also a single Qwen token.  That -- not raw vocab
  set overlap -- is what a vocab-extension decision (Epic 128) should read.

Usage:
    python3 scripts/tokenizer_alignment.py \\
        --toke-model models/toke.model \\
        --qwen-model Qwen/Qwen2.5-Coder-7B \\
        --corpus /path/to/regen_v04 --ids data/baseline_sample_ids_v04.txt \\
        --output-dir docs/alignment
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tkcanon

# ---------------------------------------------------------------------------
# Dependency checks -- hard failures, no partial mode
# ---------------------------------------------------------------------------

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - exercised by tests via monkeypatching
    print(
        "ERROR: the `transformers` package is required for tokenizer alignment "
        "(the Qwen side of the comparison).  Install it with:\n"
        "  pip install transformers\n"
        "Refusing to produce a partial analysis: the 9.8.2 partial run "
        "(qwen_vocab_size=0) produced a meaningless verdict.",
        file=sys.stderr,
    )
    sys.exit(2)

DEFAULT_TOKE_MODEL = "models/toke.model"
DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-Coder-7B"
DEFAULT_OUTPUT_DIR = "docs/alignment"
DEFAULT_SAMPLE_COUNT = 200

# ---------------------------------------------------------------------------
# Piece normalisation
# ---------------------------------------------------------------------------

SP_WORD_BOUNDARY = "▁"
SP_SPECIALS = {"<unk>", "<s>", "</s>", "<pad>"}


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 / HF ByteLevel byte -> printable-unicode mapping."""
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


_B2U = _bytes_to_unicode()
_U2B = {u: b for b, u in _B2U.items()}


def normalise_piece(piece: str, kind: str) -> str | None:
    """Map a vocabulary piece to the surface text it stands for.

    ``kind`` is ``"sentencepiece"``, ``"bytelevel"`` (HF ByteLevel alphabet) or
    ``"plain"`` (HF char-level BPE such as ``tokenizer_v03.json``).  Returns
    ``None`` for control/special pieces that have no surface form.  A byte-level
    piece that is not a complete UTF-8 sequence is returned as ``<bytes:HEX>``
    so it never spuriously equals a real string.
    """
    if kind == "sentencepiece":
        if piece in SP_SPECIALS:
            return None
        if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
            b = int(piece[3:5], 16)
            return chr(b) if b < 0x80 else f"<bytes:{b:02x}>"
        return piece.replace(SP_WORD_BOUNDARY, " ")
    if kind == "bytelevel":
        try:
            raw = bytes(_U2B[ch] for ch in piece)
        except KeyError:
            return piece  # an added/special token written in plain text
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return f"<bytes:{raw.hex()}>"
    if kind == "plain":
        return piece
    raise ValueError(f"unknown piece kind {kind!r}")


def normalise_vocab(pieces: list[str], kind: str, specials: set[str]) -> set[str]:
    out: set[str] = set()
    for p in pieces:
        if p in specials:
            continue
        n = normalise_piece(p, kind)
        if n is not None:
            out.add(n)
    return out


# ---------------------------------------------------------------------------
# Tokenizer adapters
# ---------------------------------------------------------------------------


class TokeAdapter:
    """Uniform view over a SentencePiece ``.model`` or an HF ``tokenizer.json``."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix == ".model":
            import sentencepiece as spm

            self.kind = "sentencepiece"
            self._sp = spm.SentencePieceProcessor()
            self._sp.Load(str(path))
            self.vocab_size = int(self._sp.GetPieceSize())
            self.pieces = [self._sp.IdToPiece(i) for i in range(self.vocab_size)]
            self.specials = set(SP_SPECIALS)
        else:
            from tokenizers import Tokenizer

            self._hf = Tokenizer.from_file(str(path))
            data = json.loads(path.read_text(encoding="utf-8"))
            self.kind = "bytelevel" if _uses_bytelevel(data) else "plain"
            vocab = self._hf.get_vocab()
            self.vocab_size = len(vocab)
            self.pieces = [p for p, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
            self.specials = {t["content"] for t in data.get("added_tokens", []) if t.get("special")}

    def encode_pieces(self, text: str) -> list[str]:
        if self.kind == "sentencepiece":
            return list(self._sp.encode(text, out_type=str))
        return list(self._hf.encode(text).tokens)


def _uses_bytelevel(data: dict[str, Any]) -> bool:
    pre = data.get("pre_tokenizer") or {}
    stack = [pre] + list(pre.get("pretokenizers", []))
    return any(p.get("type") == "ByteLevel" for p in stack)


def load_qwen(model_name: str, local_files_only: bool) -> Any:
    try:
        return AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    except Exception as exc:  # network / cache miss
        print(
            f"ERROR: could not load Qwen tokenizer {model_name!r}: {exc}\n"
            "If offline, make sure the model is in the HF cache (~/.cache/huggingface/hub).",
            file=sys.stderr,
        )
        sys.exit(2)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def compute_overlap(toke_norm: set[str], qwen_norm: set[str]) -> dict[str, Any]:
    inter = toke_norm & qwen_norm
    union = toke_norm | qwen_norm
    novel = toke_norm - qwen_norm
    return {
        "toke_vocab_size": len(toke_norm),
        "qwen_vocab_size": len(qwen_norm),
        "intersection_size": len(inter),
        "union_size": len(union),
        "jaccard_similarity": round(len(inter) / len(union), 6) if union else 0.0,
        "overlap_pct": round(100.0 * len(inter) / len(toke_norm), 2) if toke_norm else 0.0,
        "novel_toke_token_count": len(novel),
        "novel_pct": round(100.0 * len(novel) / len(toke_norm), 2) if toke_norm else 0.0,
        "novel_tokens_sample": sorted(novel, key=lambda s: (len(s), s))[:200],
    }


def occurrence_coverage(
    toke: TokeAdapter, qwen_norm: set[str], texts: list[str]
) -> dict[str, Any]:
    """Share of toke token occurrences whose surface text is a single Qwen token."""
    total = 0
    covered = 0
    missing: dict[str, int] = {}
    for t in texts:
        for p in toke.encode_pieces(t):
            n = normalise_piece(p, toke.kind)
            if n is None:
                continue
            total += 1
            if n in qwen_norm:
                covered += 1
            else:
                missing[n] = missing.get(n, 0) + 1
    top = sorted(missing.items(), key=lambda kv: -kv[1])[:100]
    return {
        "toke_token_occurrences": total,
        "covered_by_single_qwen_token": covered,
        "coverage_pct": round(100.0 * covered / total, 2) if total else 0.0,
        "top_uncovered": [{"piece": p, "count": c} for p, c in top],
    }


def compare_tokenization(toke: TokeAdapter, qwen: Any, recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in recs:
        tp = toke.encode_pieces(r["text"])
        qn = len(qwen.encode(r["text"], add_special_tokens=False))
        out.append({
            "task_id": r["task_id"],
            "category": r["category"],
            "chars": len(r["text"]),
            "toke_token_count": len(tp),
            "qwen_token_count": qn,
            "ratio_qwen_to_toke": round(qn / len(tp), 3) if tp else 0.0,
        })
    return out


def build_recommendation(overlap: dict[str, Any], cov: dict[str, Any], comps: list[dict[str, Any]]) -> dict[str, Any]:
    novel_pct = overlap["novel_pct"]
    cov_pct = cov["coverage_pct"]
    ratios = [c["ratio_qwen_to_toke"] for c in comps if c["ratio_qwen_to_toke"] > 0]
    avg_ratio = round(sum(ratios) / len(ratios), 3) if ratios else None
    # Occurrence coverage is the decision variable: if most toke tokens the model
    # would actually see are already single Qwen tokens, extension buys little.
    if cov_pct >= 80.0:
        action = "use_qwen_tokenizer_directly"
        feasibility = "high"
        summary = (
            f"{cov_pct:.1f}% of toke token occurrences on canonical code are already single "
            f"Qwen tokens ({novel_pct:.1f}% of toke vocab entries are novel by set overlap). "
            "Qwen's tokenizer can be used directly; extension is optional."
        )
    elif cov_pct >= 50.0:
        action = "targeted_vocab_extension"
        feasibility = "medium"
        summary = (
            f"{cov_pct:.1f}% occurrence coverage; the uncovered mass is concentrated in a few "
            "hundred toke-specific pieces (see top_uncovered). Recommend a small targeted "
            "extension rather than a full merge of the toke vocab."
        )
    else:
        action = "vocab_extension_prototype"
        feasibility = "low"
        summary = (
            f"Only {cov_pct:.1f}% of toke token occurrences are single Qwen tokens "
            f"({novel_pct:.1f}% novel vocab). Prototype a vocabulary extension."
        )
    rec = {
        "action": action,
        "decision_variable": "occurrence_coverage_pct",
        "occurrence_coverage_pct": cov_pct,
        "novel_token_pct_set_overlap": novel_pct,
        "feasibility_of_qwen_direct": feasibility,
        "summary": summary,
    }
    if avg_ratio is not None:
        rec["avg_qwen_to_toke_token_ratio"] = avg_ratio
    return rec


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def write_reports(out_dir: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / "tokenizer_alignment.json"
    jp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    o = report["overlap_stats"]
    c = report["occurrence_coverage"]
    r = report["recommendation"]
    comps = report["per_sample_comparison"]
    lines = [
        "# Tokenizer Alignment Report (toke vs Qwen2.5-Coder)",
        "",
        f"Generated: {report['generated']}  ",
        f"toke model: `{report['toke_model']}` (kind={report['toke_kind']}, sha256 `{report['toke_model_sha256'][:16]}…`)  ",
        f"Qwen model: `{report['qwen_model']}` (transformers {report['transformers_version']})  ",
        f"Sample: {len(comps)} canonical (`tkc --min` + masked) programs, seed {report['sample_seed']}",
        "",
        "## Vocabulary overlap (pieces normalised to surface text)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| toke vocab (normalised, non-special) | {o['toke_vocab_size']} |",
        f"| Qwen vocab (normalised, non-special) | {o['qwen_vocab_size']} |",
        f"| Intersection | {o['intersection_size']} |",
        f"| Jaccard | {o['jaccard_similarity']:.4f} |",
        f"| Overlap (intersection / toke vocab) | {o['overlap_pct']:.1f}% |",
        f"| Novel toke tokens | {o['novel_toke_token_count']} ({o['novel_pct']:.1f}%) |",
        "",
        "## Occurrence-weighted coverage on canonical code",
        "",
        f"- toke token occurrences: {c['toke_token_occurrences']}",
        f"- covered by a single Qwen token: {c['covered_by_single_qwen_token']} "
        f"(**{c['coverage_pct']:.1f}%**)",
        "",
        "Top uncovered toke pieces (surface text, count):",
        "",
        "```",
    ]
    for item in c["top_uncovered"][:40]:
        lines.append(f"  {item['count']:6d}  {item['piece']!r}")
    lines += ["```", "", "## Per-sample tokenization", ""]
    if comps:
        avg_t = sum(x["toke_token_count"] for x in comps) / len(comps)
        avg_q = sum(x["qwen_token_count"] for x in comps) / len(comps)
        lines += [
            f"- mean toke tokens/program: {avg_t:.1f}",
            f"- mean Qwen tokens/program: {avg_q:.1f}",
            f"- mean Qwen/toke ratio: {r.get('avg_qwen_to_toke_token_ratio', 0):.2f}x",
            "",
        ]
    lines += [
        "## Recommendation",
        "",
        f"**Action:** `{r['action']}` (decision variable: {r['decision_variable']} = {r['occurrence_coverage_pct']:.1f}%)",
        "",
        r["summary"],
        "",
        "Thresholds: coverage >= 80% -> use Qwen directly; 50-80% -> targeted extension; "
        "< 50% -> extension prototype.  Set-overlap novelty is reported for continuity with "
        "the 9.8.2 run but is not the decision variable (it counts every rare merge equally).",
        "",
    ]
    mp = out_dir / "alignment_recommendation.md"
    mp.write_text("\n".join(lines), encoding="utf-8")
    return jp, mp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="toke vs Qwen tokenizer alignment (131.20)")
    ap.add_argument("--toke-model", type=Path, default=Path(DEFAULT_TOKE_MODEL),
                    help="SentencePiece .model or HF tokenizer .json")
    ap.add_argument("--qwen-model", type=str, default=DEFAULT_QWEN_MODEL)
    ap.add_argument("--corpus", type=Path, required=True, help="regen_v04 record dir")
    ap.add_argument("--ids", type=Path, default=None, help="ids file (baseline_sample.py)")
    ap.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-mask", action="store_true", help="skip D2 string masking")
    ap.add_argument("--allow-download", action="store_true",
                    help="allow HF hub download (default: HF cache only)")
    ap.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--tkc", type=Path, default=None)
    args = ap.parse_args(argv)

    import transformers

    toke = TokeAdapter(args.toke_model)
    print(f"toke model: {args.toke_model} kind={toke.kind} vocab={toke.vocab_size}")
    qwen = load_qwen(args.qwen_model, local_files_only=not args.allow_download)
    qwen_specials = set(qwen.all_special_tokens)
    qwen_norm = normalise_vocab(list(qwen.get_vocab().keys()), "bytelevel", qwen_specials)
    toke_norm = normalise_vocab(toke.pieces, toke.kind, toke.specials)
    print(f"Qwen model: {args.qwen_model} vocab={len(qwen.get_vocab())} (normalised {len(qwen_norm)})")

    overlap = compute_overlap(toke_norm, qwen_norm)
    print(f"  overlap {overlap['overlap_pct']:.1f}%  novel {overlap['novel_pct']:.1f}%  "
          f"jaccard {overlap['jaccard_similarity']:.4f}")

    tkc = tkcanon.find_tkc(args.tkc)
    recs, fails = tkcanon.load_canonical_sample(args.corpus, args.ids, tkc, mask=not args.no_mask)
    if fails:
        print(f"  WARNING: {len(fails)} records failed --min and were dropped")
    rng = random.Random(args.seed)
    sample = rng.sample(recs, min(args.sample_count, len(recs)))
    cov = occurrence_coverage(toke, qwen_norm, [r["text"] for r in sample])
    comps = compare_tokenization(toke, qwen, sample)
    rec = build_recommendation(overlap, cov, comps)
    print(f"  occurrence coverage {cov['coverage_pct']:.1f}%  -> {rec['action']}")

    report = {
        "story": "131.20",
        "generated": datetime.now(UTC).isoformat(),
        "partial_analysis": False,
        "toke_model": str(args.toke_model),
        "toke_kind": toke.kind,
        "toke_model_sha256": toke.sha256,
        "qwen_model": args.qwen_model,
        "transformers_version": transformers.__version__,
        "tkc_version": tkcanon.tkc_version(tkc),
        "masked": not args.no_mask,
        "sample_seed": args.seed,
        "sample_source": str(args.corpus),
        "sample_ids_file": str(args.ids) if args.ids else None,
        "overlap_stats": overlap,
        "occurrence_coverage": cov,
        "recommendation": rec,
        "per_sample_comparison": comps,
    }
    jp, mp = write_reports(args.output_dir, report)
    print(f"JSON report: {jp}\nMarkdown report: {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
