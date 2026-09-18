#!/usr/bin/env python3
"""Measure the plan-D4 AddedToken substring wart for ``m= f= t= i=``.

An ``AddedToken`` (HF) or ``user_defined_symbol`` (SentencePiece) matches
ANYWHERE in the text, so an identifier ending in m/f/i/t that is immediately
followed by ``=`` -- ``xi=5``, ``sum=0``, ``buf=...``, ``cnt==1`` -- is cut as
``x`` + ``i=`` + ``5``.  Lossless, but it fragments the identifier and, worse,
buries the ``=`` inside a token that was meant to be a declaration head.

This script counts, over canonical (``tkc --min`` + D2-masked) corpus text:

* ``genuine``  -- boundary-aligned heads: ``[mfti]=`` at statement start
  (preceded by ``;`` ``{`` ``}`` or start of program).  This is where an
  AddedToken *should* fire: declaration heads plus loop steps such as
  ``;i=i+1`` (same surface, same token boundary);
* ``wart``     -- ``[a-z0-9][mfti]=``: an identifier/number character, then one
  of m/f/i/t, then ``=`` (covers ``x=`` bindings, ``==`` comparisons, and
  ``i=i+1`` loop steps where the identifier itself ends in i);
* ``realised`` -- on a given SentencePiece model with these user-defined
  symbols, how often the emitted ``[mfti]=`` piece is actually preceded by an
  identifier character (the wart as it exists in the shipped 8k model).

Recommendation rule (recorded in the report): the wart is *material* -- move
the four heads from AddedTokens to seeded merges -- when wart occurrences are
>= 5% of genuine declaration heads OR touch >= 10% of programs.

Usage:
    python3 scripts/measure_addedtoken_wart.py --corpus /path/to/regen_v04 \\
        [--ids data/baseline_sample_ids_v04.txt] [--sp-model models/toke.model] \\
        --output data/wart_d4_pre131.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tkcanon

HEADS = ("m=", "f=", "t=", "i=")
GENUINE_RX = re.compile(r"(?:^|[;{}])([mfti]=)")
WART_RX = re.compile(r"[a-z0-9]([mfti])=")
MATERIAL_RATIO = 0.05
MATERIAL_PROGRAM_SHARE = 0.10


def count_text(text: str) -> dict[str, Any]:
    genuine = Counter(m.group(1) for m in GENUINE_RX.finditer(text))
    warts: Counter[str] = Counter()
    wart_kinds: Counter[str] = Counter()
    examples: list[str] = []
    for m in WART_RX.finditer(text):
        head = m.group(1) + "="
        warts[head] += 1
        nxt = text[m.end(): m.end() + 1]
        wart_kinds["==" if nxt == "=" else "="] += 1
        if len(examples) < 3:
            s = max(0, m.start() - 6)
            examples.append(text[s: m.end() + 4])
    return {"genuine": genuine, "warts": warts, "wart_kinds": wart_kinds, "examples": examples}


def realised_on_sp(model: Path, texts: list[str]) -> dict[str, Any]:
    """How the shipped SP model (user_defined_symbols) actually cuts the text."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.Load(str(model))
    emitted: Counter[str] = Counter()
    mid_ident: Counter[str] = Counter()
    ident_rx = re.compile(r"[a-z0-9]$")
    for t in texts:
        pieces = sp.encode(t, out_type=str)
        for i, p in enumerate(pieces):
            if p in HEADS:
                emitted[p] += 1
                prev = pieces[i - 1].replace("▁", " ") if i else ""
                if ident_rx.search(prev):
                    mid_ident[p] += 1
    tot_e = sum(emitted.values())
    tot_m = sum(mid_ident.values())
    return {
        "model": str(model),
        "head_pieces_emitted": dict(emitted),
        "head_pieces_preceded_by_identifier_char": dict(mid_ident),
        "total_emitted": tot_e,
        "total_mid_identifier": tot_m,
        "mid_identifier_pct": round(100.0 * tot_m / tot_e, 2) if tot_e else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D4 AddedToken substring wart (131.20)")
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--ids", type=Path, default=None, help="ids file; default: whole corpus")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sp-model", type=Path, action="append", default=[],
                    help="SentencePiece model(s) with m=/f=/t=/i= user symbols to measure realised wart")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--tkc", type=Path, default=None)
    args = ap.parse_args(argv)

    tkc = tkcanon.find_tkc(args.tkc)
    recs, fails = tkcanon.load_canonical_sample(args.corpus, args.ids, tkc, mask=True, limit=args.limit)
    texts = [r["text"] for r in recs]
    print(f"canonical programs: {len(texts)} ({len(fails)} failed --min)")

    genuine: Counter[str] = Counter()
    warts: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    per_cat: dict[str, Counter[str]] = {}
    programs_with_wart = 0
    examples: list[str] = []
    total_chars = 0
    for r in recs:
        c = count_text(r["text"])
        genuine.update(c["genuine"])
        warts.update(c["warts"])
        kinds.update(c["wart_kinds"])
        total_chars += len(r["text"])
        pc = per_cat.setdefault(r["category"], Counter())
        pc["genuine"] += sum(c["genuine"].values())
        pc["wart"] += sum(c["warts"].values())
        pc["programs"] += 1
        if c["warts"]:
            programs_with_wart += 1
            pc["programs_with_wart"] += 1
            if len(examples) < 12:
                examples.extend(c["examples"][:1])

    g_tot = sum(genuine.values())
    w_tot = sum(warts.values())
    ratio = w_tot / g_tot if g_tot else 0.0
    share = programs_with_wart / len(recs) if recs else 0.0
    material = ratio >= MATERIAL_RATIO or share >= MATERIAL_PROGRAM_SHARE
    per_head = {
        h: {"genuine": genuine[h], "wart": warts[h],
            "wart_per_genuine_pct": round(100.0 * warts[h] / genuine[h], 2) if genuine[h] else None}
        for h in HEADS
    }
    realised = [realised_on_sp(m, texts) for m in args.sp_model]

    print(f"genuine decl heads: {g_tot}   wart occurrences: {w_tot}   ratio {100 * ratio:.2f}%")
    print(f"programs with >=1 wart: {programs_with_wart}/{len(recs)} ({100 * share:.1f}%)   "
          f"warts per 1k chars: {1000 * w_tot / total_chars:.3f}")
    for h in HEADS:
        print(f"  {h}: genuine={per_head[h]['genuine']:7d} wart={per_head[h]['wart']:6d} "
              f"({per_head[h]['wart_per_genuine_pct']}%)")
    print(f"  wart followed by: {dict(kinds)}")
    for r in realised:
        print(f"  realised on {r['model']}: {r['total_mid_identifier']}/{r['total_emitted']} "
              f"head pieces preceded by identifier char ({r['mid_identifier_pct']}%)")
    rec = ("seeded_merges" if material else "added_tokens")
    print(f"RECOMMENDATION: {rec} (material={material}; rule: wart/genuine >= {MATERIAL_RATIO:.0%} "
          f"or programs touched >= {MATERIAL_PROGRAM_SHARE:.0%})")

    report = {
        "story": "131.20", "generated": datetime.now(UTC).isoformat(),
        "tkc_version": tkcanon.tkc_version(tkc), "corpus": str(args.corpus),
        "ids": str(args.ids) if args.ids else None, "num_programs": len(recs),
        "min_failures": fails, "total_chars": total_chars,
        "genuine_decl_heads": g_tot, "wart_occurrences": w_tot,
        "wart_per_genuine_pct": round(100 * ratio, 3),
        "programs_with_wart": programs_with_wart, "programs_with_wart_pct": round(100 * share, 2),
        "warts_per_1k_chars": round(1000 * w_tot / total_chars, 4) if total_chars else None,
        "per_head": per_head, "wart_followed_by": dict(kinds),
        "per_category": {k: dict(v) for k, v in sorted(per_cat.items())},
        "examples": examples, "realised_on_sp_models": realised,
        "rule": {"material_ratio": MATERIAL_RATIO, "material_program_share": MATERIAL_PROGRAM_SHARE},
        "material": material, "recommendation": rec,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
