#!/usr/bin/env python3
"""Single-token coverage of v0.4 toke syntax for one or more tokenizers.

Story 131.20 (plan Phase 0 item 3) rewrite of the 11.4.6 script: v0.4
``KEY_PATTERNS``, a real CLI, SentencePiece *and* HF-JSON models, and an
in-context measurement on canonical corpus text (``tkc --min`` + D2 masking)
in addition to the standalone "does the pattern alone encode to one piece"
check.  The in-context numbers are the ones that matter:

* ``unsplit``  -- the occurrence lies entirely inside ONE token (it may be part
  of a bigger merge such as ``):i64{`` -- that is fine, plan D4 calls fragment
  merges a feature);
* ``exact``    -- the occurrence IS a token on its own.

READ-ONLY: never modifies a model.

Usage:
    python3 scripts/eval_syntax_tokens.py \\
        --model models/toke.model --model models/32k/toke.model --model tokenizer_v03.json \\
        --corpus /path/to/regen_v04 --ids data/baseline_sample_ids_v04.txt \\
        --output docs/eval_syntax_tokens_v04_pre131.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tkcanon

# ---------------------------------------------------------------------------
# v0.4 key patterns (plan D4 forced list + verification-only groups)
# ---------------------------------------------------------------------------

# Each entry: (surface text, in-context regex).  The regex keeps the standalone
# text as the match so spans line up with token offsets.
Pattern = tuple[str, str]

KEY_PATTERNS: dict[str, list[Pattern]] = {
    # --- D4 forced list -------------------------------------------------
    "operators": [
        ("==", r"=="), ("!=", r"!="), ("&&", r"&&"), ("||", r"\|\|"),
        ("<=", r"<="), (">=", r">="),
    ],
    "expr_if": [("if(", r"\bif\("), ("el{", r"\bel\{")],
    "match": [("mt ", r"\bmt "), ("$ok:", r"\$ok:"), ("$err:", r"\$err:")],
    "interpolation_open": [("\\(", r"\\\(")],
    "collection_open": [("@(", r"@\(")],
    "type_sigils": [
        ("$i64", r"\$i64"), ("$f64", r"\$f64"), ("$str", r"\$str"),
        ("$bool", r"\$bool"), ("$u64", r"\$u64"), ("$byte", r"\$byte"),
    ],
    "decl_heads": [
        ("m=", r"(?<![a-z0-9])m="), ("f=", r"(?<![a-z0-9])f="),
        ("t=", r"(?<![a-z0-9])t="), ("i=", r"(?<![a-z0-9])i="),
    ],
    "loop": [("lp(", r"\blp\(")],
    "early_return": [("{<", r"\{<"), (";<", r";<")],
    # --- verification-only (NOT forced; must merge naturally, plan D4) ----
    "type_annotations": [
        (":i64", r":i64\b"), (":str", r":str\b"), (":bool", r":bool\b"),
        (":f64", r":f64\b"), (":u64", r":u64\b"), ("@i64", r"@i64\b"), ("@str", r"@str\b"),
    ],
    "stdlib_surface": [
        (".get(", r"\.get\("), (".len", r"\.len\b"), ("std.io", r"std\.io\b"),
        ("io.println(", r"io\.println\("), ("let ", r"\blet "), ("mut.", r"\bmut\."),
    ],
}

FORCED_GROUPS = (
    "operators", "expr_if", "match", "interpolation_open", "collection_open",
    "type_sigils", "decl_heads", "loop", "early_return",
)

# ---------------------------------------------------------------------------
# Tokenizer adapters (pieces + char spans)
# ---------------------------------------------------------------------------


class Adapter:
    """``spans(text)`` -> list of (piece, start, end) char offsets covering ``text``."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.name = path.name if path.name != "toke.model" else f"{path.parent.name}/{path.name}"
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix == ".model":
            import sentencepiece as spm

            self.kind = "sentencepiece"
            self._sp = spm.SentencePieceProcessor()
            self._sp.Load(str(path))
            self.vocab_size = int(self._sp.GetPieceSize())
        else:
            from tokenizers import Tokenizer

            self.kind = "hf"
            self._hf = Tokenizer.from_file(str(path))
            self.vocab_size = int(self._hf.get_vocab_size())

    def pieces(self, text: str) -> list[str]:
        if self.kind == "sentencepiece":
            return list(self._sp.encode(text, out_type=str))
        return list(self._hf.encode(text).tokens)

    def spans(self, text: str) -> list[tuple[str, int, int]]:
        if self.kind == "hf":
            enc = self._hf.encode(text)
            return [(t, s, e) for t, (s, e) in zip(enc.tokens, enc.offsets, strict=True)]
        proto = self._sp.encode(text, out_type="immutable_proto")
        raw = text.encode("utf-8")
        out: list[tuple[str, int, int]] = []
        for p in proto.pieces:
            s = len(raw[: p.begin].decode("utf-8", errors="ignore"))
            e = len(raw[: p.end].decode("utf-8", errors="ignore"))
            out.append((p.piece, s, e))
        return out


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


def _surface(piece: str) -> str:
    """Surface text of a SentencePiece piece (``▁`` -> space, ``<0xNN>`` -> byte)."""
    if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
        return chr(int(piece[3:5], 16))
    return piece.replace("▁", " ")


def standalone(adapter: Adapter, patterns: dict[str, list[Pattern]]) -> dict[str, list[dict[str, Any]]]:
    """Pattern alone -> pieces.  SentencePiece models trained with
    ``add_dummy_prefix`` (the 32k model) prepend ``▁``; that leading marker is
    stripped so the count reflects the pattern, not the prefix.  ``lossy`` is
    set when the pieces do not reproduce the pattern (chars were dropped)."""
    out: dict[str, list[dict[str, Any]]] = {}
    for group, pats in patterns.items():
        rows = []
        for surface, _ in pats:
            pieces = adapter.pieces(surface)
            if adapter.kind == "sentencepiece" and pieces:
                if pieces[0] == "▁":
                    pieces = pieces[1:]
                elif pieces[0].startswith("▁") and not surface.startswith(" "):
                    pieces = [pieces[0][1:]] + pieces[1:]
            joined = "".join(_surface(p) for p in pieces) if adapter.kind == "sentencepiece" else "".join(pieces)
            lossy = joined != surface
            rows.append({"pattern": surface, "pieces": pieces, "num_pieces": len(pieces),
                         "is_single_token": len(pieces) == 1 and not lossy, "lossy": lossy})
        out[group] = rows
    return out


def in_context(
    adapter: Adapter, texts: list[str], patterns: dict[str, list[Pattern]]
) -> dict[str, list[dict[str, Any]]]:
    compiled = {g: [(s, re.compile(rx)) for s, rx in pats] for g, pats in patterns.items()}
    counts: dict[tuple[str, str], list[int]] = {
        (g, s): [0, 0, 0] for g, pats in patterns.items() for s, _ in pats
    }  # occurrences, unsplit, exact
    for text in texts:
        spans = adapter.spans(text)
        # token index by start offset for a quick "which token covers char c"
        cover = [-1] * (len(text) + 1)
        for ti, (_, s, e) in enumerate(spans):
            for c in range(s, e):
                cover[c] = ti
        for g, cpats in compiled.items():
            for surface, rx in cpats:
                for m in rx.finditer(text):
                    s, e = m.start(), m.end()
                    tally = counts[(g, surface)]
                    tally[0] += 1
                    toks = {cover[i] for i in range(s, e)}
                    if len(toks) == 1 and -1 not in toks:
                        tally[1] += 1
                        ti = next(iter(toks))
                        if spans[ti][1] == s and spans[ti][2] == e:
                            tally[2] += 1
    out: dict[str, list[dict[str, Any]]] = {}
    for g, pats in patterns.items():
        rows = []
        for surface, _ in pats:
            occ, uns, ex = counts[(g, surface)]
            rows.append({
                "pattern": surface, "occurrences": occ,
                "unsplit": uns, "unsplit_pct": round(100.0 * uns / occ, 2) if occ else None,
                "exact": ex, "exact_pct": round(100.0 * ex / occ, 2) if occ else None,
            })
        out[g] = rows
    return out


def summarise(sa: dict[str, list[dict[str, Any]]], ic: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    forced_single = forced_total = 0
    occ = uns = ex = 0
    for g in FORCED_GROUPS:
        for r in sa[g]:
            forced_total += 1
            forced_single += int(r["is_single_token"])
        for r in ic[g]:
            occ += r["occurrences"]
            uns += r["unsplit"]
            ex += r["exact"]
    return {
        "forced_patterns_single_token_standalone": f"{forced_single}/{forced_total}",
        "forced_patterns_single_token_standalone_pct": round(100.0 * forced_single / forced_total, 1),
        "forced_occurrences_in_sample": occ,
        "forced_unsplit_pct": round(100.0 * uns / occ, 2) if occ else None,
        "forced_exact_pct": round(100.0 * ex / occ, 2) if occ else None,
        "split_occurrences": occ - uns,
    }


def pattern_frequency(texts: list[str], patterns: dict[str, list[Pattern]]) -> dict[str, int]:
    freq: dict[str, int] = {}
    for pats in patterns.values():
        for surface, rx in pats:
            freq[surface] = sum(len(re.findall(rx, t)) for t in texts)
    return freq


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def print_model_report(name: str, sa: dict[str, list[dict[str, Any]]], ic: dict[str, list[dict[str, Any]]], summ: dict[str, Any]) -> None:
    print(f"\n{'=' * 72}\n  {name}\n{'=' * 72}")
    for g in KEY_PATTERNS:
        tag = "forced" if g in FORCED_GROUPS else "verify"
        print(f"  [{tag}] {g}")
        for a, b in zip(sa[g], ic[g], strict=True):
            st = "SINGLE" if a["is_single_token"] else ("LOSSY" if a["lossy"] else f"{a['num_pieces']} pieces")
            u = f"{b['unsplit_pct']:6.1f}%" if b["unsplit_pct"] is not None else "   n/a "
            e = f"{b['exact_pct']:6.1f}%" if b["exact_pct"] is not None else "   n/a "
            print(f"    {a['pattern']!r:14} standalone={st:10} occ={b['occurrences']:6d} "
                  f"unsplit={u} exact={e}  {a['pieces']}")
    print(f"  forced standalone single-token: {summ['forced_patterns_single_token_standalone']} "
          f"({summ['forced_patterns_single_token_standalone_pct']}%)")
    print(f"  forced in-context unsplit: {summ['forced_unsplit_pct']}%  exact: {summ['forced_exact_pct']}%  "
          f"(split occurrences: {summ['split_occurrences']} / {summ['forced_occurrences_in_sample']})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="v0.4 syntax single-token coverage (131.20)")
    ap.add_argument("--model", type=Path, action="append", required=True,
                    help="SentencePiece .model or HF tokenizer .json (repeatable)")
    ap.add_argument("--corpus", type=Path, required=True, help="regen_v04 record dir")
    ap.add_argument("--ids", type=Path, default=None, help="ids file (baseline_sample.py)")
    ap.add_argument("--limit", type=int, default=None, help="max records (default: all in ids/corpus)")
    ap.add_argument("--no-mask", action="store_true", help="skip D2 string masking")
    ap.add_argument("--output", type=Path, default=None, help="JSON report path")
    ap.add_argument("--tkc", type=Path, default=None)
    args = ap.parse_args(argv)

    for m in args.model:
        if not m.is_file():
            print(f"ERROR: model not found: {m}", file=sys.stderr)
            return 1
    tkc = tkcanon.find_tkc(args.tkc)
    recs, fails = tkcanon.load_canonical_sample(args.corpus, args.ids, tkc, mask=not args.no_mask,
                                                limit=args.limit)
    texts = [r["text"] for r in recs]
    print(f"canonical sample: {len(texts)} programs ({len(fails)} failed --min), "
          f"{sum(len(t) for t in texts)} chars, masked={not args.no_mask}")
    freq = pattern_frequency(texts, KEY_PATTERNS)

    models: dict[str, Any] = {}
    for mp in args.model:
        ad = Adapter(mp)
        sa = standalone(ad, KEY_PATTERNS)
        ic = in_context(ad, texts, KEY_PATTERNS)
        summ = summarise(sa, ic)
        print_model_report(ad.name, sa, ic, summ)
        models[str(mp)] = {
            "kind": ad.kind, "vocab_size": ad.vocab_size, "sha256": ad.sha256,
            "standalone": sa, "in_context": ic, "summary": summ,
        }

    report = {
        "story": "131.20",
        "title": "v0.4 syntax single-token coverage (pre-rewrite)",
        "generated": datetime.now(UTC).isoformat(),
        "tkc_version": tkcanon.tkc_version(tkc),
        "corpus": str(args.corpus), "ids": str(args.ids) if args.ids else None,
        "num_programs": len(texts), "masked": not args.no_mask,
        "pattern_frequency_in_sample": freq,
        "forced_groups": list(FORCED_GROUPS),
        "models": models,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nReport written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
