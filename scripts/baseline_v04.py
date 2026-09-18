#!/usr/bin/env python3
"""TEMSpec baseline over the stratified v0.4 sample (story 131.20, plan D7).

Measures every available tokenizer on IDENTICAL normalised text -- ``tkc --min``
canonical form (plan D6) with string bodies masked to ``_`` (plan D2) -- and
writes:

* ``data/baseline_v04_<label>.json``            full results
* ``data/baseline_v04_<label>_per_record.csv``  TEMSpec §5.3 per-task counts
* ``docs/baseline_v04_<label>.md``              tables + methodology

Tokenizers (TEMSpec §3; a missing one is reported as skipped, never faked):
``tokenizer_v03`` (HF json), ``sp8k`` (``models/toke.model``), ``sp32k``
(``models/32k/toke.model``), ``qwen2.5-coder`` (HF cache), ``cl100k_base`` and
``o200k_base`` (tiktoken; o200k informational), ``llama3`` (HF cache only).

Metrics per tokenizer: total tokens; tokens/program mean, median, p95;
fertility = tokens/char (TEMSpec §2.4, mean of per-program and corpus-level);
vocab utilisation (§2.5); ratio of total tokens vs cl100k_base (§2.2 form);
unmasked fertility (informational, D2); roundtrip fidelity; per-category and
per-difficulty breakdowns; 10,000-resample percentile bootstrap 95% CIs (§5.2)
on mean tokens/program, mean fertility and the vs-cl100k ratio.

Re-run for the post-rewrite comparison (131.21) with the SAME ids file:

    python3 scripts/baseline_v04.py --ids data/baseline_sample_ids_v04.txt \\
        --corpus <dir> --label post131 [--model name=path ...]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tkcanon

REPO = Path(__file__).resolve().parent.parent
BOOTSTRAP_N = 10_000
BASE_TOKENIZER = "cl100k_base"


# ---------------------------------------------------------------------------
# Tokenizer registry
# ---------------------------------------------------------------------------


class Tok:
    name: str
    kind: str
    vocab_size: int
    version: str
    ident: str  # sha256 of file or model id

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError


class SPTok(Tok):
    def __init__(self, name: str, path: Path) -> None:
        import sentencepiece as spm

        self.name, self.kind = name, "sentencepiece"
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(str(path))
        self.vocab_size = int(self._sp.GetPieceSize())
        self.version = f"sentencepiece {spm.__version__}"
        self.ident = f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
        self.unk_id = int(self._sp.unk_id())

    def encode(self, text: str) -> list[int]:
        return list(self._sp.encode(text, out_type=int))

    def decode(self, ids: list[int]) -> str:
        return str(self._sp.decode(ids))

    def unk_chars(self, text: str) -> list[str]:
        """Surface text of every piece that encoded to ``<unk>``."""
        proto = self._sp.encode(text, out_type="immutable_proto")
        return [p.surface for p in proto.pieces if p.id == self.unk_id]


class HFJsonTok(Tok):
    def __init__(self, name: str, path: Path) -> None:
        import tokenizers
        from tokenizers import Tokenizer

        self.name, self.kind = name, "hf-json"
        self._t = Tokenizer.from_file(str(path))
        self.vocab_size = int(self._t.get_vocab_size())
        self.version = f"tokenizers {tokenizers.__version__}"
        self.ident = f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
        data = json.loads(path.read_text(encoding="utf-8"))
        # A file with no decoder (tokenizer_v03.json) is char-level: HF's
        # generic decode joins pieces with spaces, so roundtrip is checked by
        # plain concatenation instead.  unk_token=None means unknown chars are
        # silently DROPPED -- that is what ``uncovered_chars`` counts.
        self._has_decoder = data.get("decoder") is not None
        self.unk_token = data.get("model", {}).get("unk_token")

    def encode(self, text: str) -> list[int]:
        return list(self._t.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: list[int]) -> str:
        if self._has_decoder:
            return str(self._t.decode(ids, skip_special_tokens=True))
        return "".join(self._t.id_to_token(i) or "" for i in ids)

    def uncovered_chars(self, text: str) -> list[str]:
        """Characters of ``text`` the tokenizer dropped.

        Multiset difference between the input and the decoded output: exact for
        a tokenizer that only ever drops characters (``unk_token: null``).  HF
        offsets are NOT used -- for a char-level BPE they stretch neighbouring
        tokens over dropped characters and mis-attribute the loss.
        """
        decoded = self.decode(self.encode(text))
        diff = Counter(text) - Counter(decoded)
        return list(diff.elements())


class HFHubTok(Tok):
    def __init__(self, name: str, model_id: str, allow_download: bool) -> None:
        import transformers
        from transformers import AutoTokenizer

        self.name, self.kind = name, "hf-hub"
        self._t = AutoTokenizer.from_pretrained(model_id, local_files_only=not allow_download)
        self.vocab_size = len(self._t)
        self.version = f"transformers {transformers.__version__}"
        self.ident = model_id

    def encode(self, text: str) -> list[int]:
        # TEMSpec §3.3: default settings.  Qwen2 adds no BOS/EOS by default;
        # add_special_tokens is left at its default deliberately.
        return list(self._t.encode(text))

    def decode(self, ids: list[int]) -> str:
        return str(self._t.decode(ids, skip_special_tokens=True))


class TiktokenTok(Tok):
    def __init__(self, name: str, encoding: str) -> None:
        import tiktoken

        self.name, self.kind = name, "tiktoken"
        self._e = tiktoken.get_encoding(encoding)
        self.vocab_size = int(self._e.n_vocab)
        self.version = f"tiktoken {tiktoken.__version__}"
        self.ident = encoding

    def encode(self, text: str) -> list[int]:
        return list(self._e.encode(text))

    def decode(self, ids: list[int]) -> str:
        return str(self._e.decode(ids))


DEFAULT_MODELS: list[tuple[str, str, str]] = [
    # name, kind, locator
    ("tokenizer_v03", "hf-json", str(REPO / "tokenizer_v03.json")),
    ("sp8k", "sp", str(REPO / "models" / "toke.model")),
    ("sp32k", "sp", str(REPO / "models" / "32k" / "toke.model")),
    ("qwen2.5-coder", "hf-hub", "Qwen/Qwen2.5-Coder-7B"),
    ("cl100k_base", "tiktoken", "cl100k_base"),
    ("o200k_base", "tiktoken", "o200k_base"),
    ("llama3", "hf-hub", "meta-llama/Meta-Llama-3-8B"),
]


def load_tokenizers(specs: list[tuple[str, str, str]], allow_download: bool) -> tuple[list[Tok], list[dict[str, str]]]:
    toks: list[Tok] = []
    skipped: list[dict[str, str]] = []
    for name, kind, loc in specs:
        try:
            if kind == "sp":
                toks.append(SPTok(name, Path(loc)))
            elif kind == "hf-json":
                toks.append(HFJsonTok(name, Path(loc)))
            elif kind == "hf-hub":
                toks.append(HFHubTok(name, loc, allow_download))
            elif kind == "tiktoken":
                toks.append(TiktokenTok(name, loc))
            else:
                raise ValueError(f"unknown tokenizer kind {kind}")
            print(f"  loaded {name:14s} {kind:9s} vocab={toks[-1].vocab_size}")
        except Exception as exc:  # missing file / not in cache / gated
            reason = str(exc).splitlines()[0][:200]
            skipped.append({"name": name, "kind": kind, "locator": loc, "reason": reason})
            print(f"  SKIPPED {name}: {reason}")
    return toks, skipped


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def bootstrap_ci(values: np.ndarray, stat: str = "mean", n: int = BOOTSTRAP_N, seed: int = 131) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    samples = values[idx]
    est = samples.mean(axis=1) if stat == "mean" else np.median(samples, axis=1)
    lo, hi = np.percentile(est, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_ratio_ci(num: np.ndarray, den: np.ndarray, n: int = BOOTSTRAP_N, seed: int = 131) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n, len(num)))
    r = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    lo, hi = np.percentile(r, [2.5, 97.5])
    return float(lo), float(hi)


def p95(xs: list[int]) -> float:
    s = sorted(xs)
    return float(s[min(int(len(s) * 0.95), len(s) - 1)]) if s else 0.0


def summarise(counts: list[int], chars: list[int]) -> dict[str, Any]:
    c = np.array(counts, dtype=float)
    ch = np.array(chars, dtype=float)
    fert = c / np.where(ch == 0, 1, ch)
    return {
        "n": len(counts),
        "total_tokens": int(c.sum()),
        "total_chars": int(ch.sum()),
        "tokens_per_program_mean": float(c.mean()),
        "tokens_per_program_median": float(np.median(c)),
        "tokens_per_program_p95": p95(counts),
        "fertility_mean": float(fert.mean()),
        "fertility_corpus": float(c.sum() / ch.sum()) if ch.sum() else 0.0,
        "chars_per_token": float(ch.sum() / c.sum()) if c.sum() else 0.0,
    }


# ---------------------------------------------------------------------------
# Main measurement
# ---------------------------------------------------------------------------


def measure(tok: Tok, recs: list[dict[str, Any]]) -> dict[str, Any]:
    counts: list[int] = []
    counts_unmasked: list[int] = []
    used: set[int] = set()
    roundtrip_ok = 0
    unk = 0
    lossy: dict[str, int] = defaultdict(int)  # dropped / unk surface -> count
    for r in recs:
        ids = tok.encode(r["text"])
        counts.append(len(ids))
        used.update(ids)
        if isinstance(tok, SPTok):
            unk += ids.count(tok.unk_id)
            for ch in tok.unk_chars(r["text"]):
                lossy[ch] += 1
        elif isinstance(tok, HFJsonTok):
            for ch in tok.uncovered_chars(r["text"]):
                lossy[ch] += 1
        try:
            if tok.decode(ids) == r["text"]:
                roundtrip_ok += 1
        except Exception:
            pass
        counts_unmasked.append(len(tok.encode(r["min"])))
    chars = [len(r["text"]) for r in recs]
    chars_un = [len(r["min"]) for r in recs]
    out: dict[str, Any] = {
        "kind": tok.kind, "version": tok.version, "ident": tok.ident, "vocab_size": tok.vocab_size,
        "masked": summarise(counts, chars),
        "unmasked_informational": summarise(counts_unmasked, chars_un),
        "vocab_utilization": len(used) / tok.vocab_size if tok.vocab_size else 0.0,
        "unique_tokens_used": len(used),
        "roundtrip_ok": roundtrip_ok, "roundtrip_pct": round(100.0 * roundtrip_ok / len(recs), 2),
        "unk_tokens": unk,
        "lossy_chars_total": int(sum(lossy.values())),
        "lossy_chars_top": sorted(lossy.items(), key=lambda kv: -kv[1])[:12],
        "per_record_counts": counts,
    }
    arr_c = np.array(counts, dtype=float)
    arr_ch = np.array(chars, dtype=float)
    out["ci95"] = {
        "tokens_per_program_mean": bootstrap_ci(arr_c, "mean"),
        "tokens_per_program_median": bootstrap_ci(arr_c, "median"),
        "fertility_mean": bootstrap_ci(arr_c / arr_ch, "mean"),
    }
    by_cat: dict[str, list[int]] = defaultdict(list)
    by_cat_ch: dict[str, list[int]] = defaultdict(list)
    by_diff: dict[str, list[int]] = defaultdict(list)
    by_diff_ch: dict[str, list[int]] = defaultdict(list)
    for r, n in zip(recs, counts, strict=True):
        by_cat[r["category"]].append(n)
        by_cat_ch[r["category"]].append(len(r["text"]))
        by_diff[str(r["difficulty"])].append(n)
        by_diff_ch[str(r["difficulty"])].append(len(r["text"]))
    out["per_category"] = {k: summarise(by_cat[k], by_cat_ch[k]) for k in sorted(by_cat)}
    out["per_difficulty"] = {k: summarise(by_diff[k], by_diff_ch[k]) for k in sorted(by_diff)}
    return out


def add_ratios(results: dict[str, Any], base: str) -> None:
    if base not in results:
        return
    b = np.array(results[base]["per_record_counts"], dtype=float)
    for name, r in results.items():
        c = np.array(r["per_record_counts"], dtype=float)
        ratio = float(c.sum() / b.sum())
        r["vs_" + base] = {
            "ratio_total_tokens": ratio,
            "reduction_pct": 100.0 * (1.0 - ratio),
            "ci95_ratio": bootstrap_ratio_ci(c, b),
            "per_category_ratio": {
                k: r["per_category"][k]["total_tokens"] / results[base]["per_category"][k]["total_tokens"]
                for k in r["per_category"]
            },
        }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_csv(path: Path, recs: list[dict[str, Any]], results: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "tokenizer", "language", "token_count", "char_count", "category", "difficulty"])
        for name, r in results.items():
            for rec, n in zip(recs, r["per_record_counts"], strict=True):
                w.writerow([rec["task_id"], name, "toke", n, len(rec["text"]), rec["category"], rec["difficulty"]])


def fmt_ci(ci: tuple[float, float], nd: int = 1) -> str:
    return f"[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    res = report["results"]
    order = [n for n in ("tokenizer_v03", "sp8k", "sp32k", "qwen2.5-coder", "cl100k_base", "o200k_base", "llama3") if n in res]
    order += [n for n in res if n not in order]
    m = report["meta"]
    L: list[str] = []
    L += [f"# Tokenizer baseline v0.4 — `{m['label']}`", "",
          f"Story 131.20 (plan D7). Generated {m['generated']}. **This is the PRE-rewrite baseline**; "
          "131.21 re-runs the same script with `--label post131` over the same ids." if m["label"] == "pre131"
          else f"Story 131.21 re-run of the 131.20 baseline. Generated {m['generated']}.", ""]
    L += ["## Headline (masked canonical text)", "",
          "| tokenizer | vocab | total tokens | tokens/program mean [95% CI] | median | p95 | fertility tok/char (mean) [95% CI] | chars/token | vocab util. | vs cl100k ratio [95% CI] | roundtrip | lossy chars |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in order:
        r = res[n]
        mk = r["masked"]
        ci = r["ci95"]
        vs = r.get("vs_cl100k_base")
        vs_s = f"{vs['ratio_total_tokens']:.4f} {fmt_ci(vs['ci95_ratio'], 4)}" if vs else "—"
        L.append(f"| {n} | {r['vocab_size']} | {mk['total_tokens']:,} | {mk['tokens_per_program_mean']:.1f} {fmt_ci(ci['tokens_per_program_mean'])} "
                 f"| {mk['tokens_per_program_median']:.0f} | {mk['tokens_per_program_p95']:.0f} | {mk['fertility_mean']:.4f} {fmt_ci(ci['fertility_mean'], 4)} "
                 f"| {mk['chars_per_token']:.3f} | {100 * r['vocab_utilization']:.1f}% | {vs_s} | {r['roundtrip_pct']:.1f}% | {r['lossy_chars_total']:,} |")
    L += ["", "Ratio < 1 means fewer tokens than cl100k_base on the same text (TEMSpec §2.2 compression-ratio form; "
          "same source language, different tokenizers — informational cross-tokenizer density, §6.2, not the §6.1 gate metric). "
          "`lossy chars` = characters dropped (HF file with `unk_token: null`) or mapped to `<unk>` (SentencePiece without "
          "byte fallback); a tokenizer with lossy chars > 0 under-counts and its row is informational only.", ""]
    lossy_rows = [(n, res[n]["lossy_chars_top"]) for n in order if res[n]["lossy_chars_total"]]
    if lossy_rows:
        L += ["Lossy tokenizers — most frequent dropped/unk surfaces (surface, count):", ""]
        for n, top in lossy_rows:
            L.append(f"- **{n}**: " + ", ".join(f"`{ch!r}`×{c}" for ch, c in top))
        L.append("")
    L += ["## Unmasked (informational, plan D2)", "",
          "| tokenizer | total tokens | tokens/program mean | fertility (mean) | chars/token |", "|---|---:|---:|---:|---:|"]
    for n in order:
        u = res[n]["unmasked_informational"]
        L.append(f"| {n} | {u['total_tokens']:,} | {u['tokens_per_program_mean']:.1f} | {u['fertility_mean']:.4f} | {u['chars_per_token']:.3f} |")
    L += ["", "## Per-category (masked): tokens/program mean", ""]
    cats = sorted(next(iter(res.values()))["per_category"])
    L.append("| category | n | " + " | ".join(order) + " |")
    L.append("|---|---:|" + "|".join("---:" for _ in order) + "|")
    for c in cats:
        n0 = res[order[0]]["per_category"][c]["n"]
        L.append(f"| {c} | {n0} | " + " | ".join(f"{res[n]['per_category'][c]['tokens_per_program_mean']:.1f}" for n in order) + " |")
    L += ["", "## Per-category (masked): fertility (corpus tokens/char)", ""]
    L.append("| category | " + " | ".join(order) + " |")
    L.append("|---|" + "|".join("---:" for _ in order) + "|")
    for c in cats:
        L.append(f"| {c} | " + " | ".join(f"{res[n]['per_category'][c]['fertility_corpus']:.4f}" for n in order) + " |")
    L += ["", "## Per-difficulty (masked): tokens/program mean", ""]
    diffs = sorted(next(iter(res.values()))["per_difficulty"])
    L.append("| difficulty | n | " + " | ".join(order) + " |")
    L.append("|---|---:|" + "|".join("---:" for _ in order) + "|")
    for d in diffs:
        n0 = res[order[0]]["per_difficulty"][d]["n"]
        L.append(f"| {d} | {n0} | " + " | ".join(f"{res[n]['per_difficulty'][d]['tokens_per_program_mean']:.1f}" for n in order) + " |")
    L += syntax_section(report)
    L += wart_section(report)
    L += ["", "## Tokenizers", "", "| name | kind | version | identity |", "|---|---|---|---|"]
    for n in order:
        r = res[n]
        L.append(f"| {n} | {r['kind']} | {r['version']} | `{r['ident']}` |")
    for s in report["skipped_tokenizers"]:
        L.append(f"| {s['name']} | {s['kind']} | — | SKIPPED: {s['reason']} |")
    L += ["", "## Methodology", "",
          f"- Sample: {m['n_records']} records, ids file `{m['ids_file']}` (sha256 `{m['ids_file_sha256']}`), "
          f"stratified by category × difficulty, seed 131, drawn from the freeze-129 `MANIFEST.jsonl`.",
          f"- Corpus source: `{m['corpus']}` ({m['corpus_note']}).",
          f"- Record integrity: {m['record_sha_matches']}/{m['n_records']} record files match the freeze-129 sha256 in the ids file"
          + (f"; MISMATCHES: {m['record_sha_mismatches'][:10]}" if m["record_sha_mismatches"] else "") + ".",
          f"- Records dropped (failed `tkc --min`): {len(m['failures'])}.",
          f"- Canonical form: `tkc --min` ({m['tkc_version']}), one program per line; strings masked to `_` per plan D2 "
          f"(`toke/scripts/patterns/mask_strings.py`, keeps `\\(...)` interpolation interiors).",
          f"- sample sha256 (sha256 over the sorted per-record `min_sha256`s): `{m['sample_min_sha256']}`; "
          f"masked-text sha256: `{m['sample_masked_sha256']}`.",
          "- Token counts: `len(encode(text))` with default settings (TEMSpec §3.3); Qwen adds no BOS/EOS by default.",
          f"- CIs: percentile bootstrap, {BOOTSTRAP_N:,} resamples, seed 131 (TEMSpec §5.2). Fertility = tokens/char (§2.4). "
          "Vocab utilisation = unique ids used / vocab size (§2.5).",
          f"- Per-record counts: `{m['csv_file']}` (TEMSpec §5.3).", ""]
    if report.get("notes_md"):
        L += ["", report["notes_md"].rstrip(), ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


def syntax_section(report: dict[str, Any]) -> list[str]:
    """v0.4 syntax gap table from ``docs/eval_syntax_tokens_v04_<label>.json`` if present."""
    p = REPO / "docs" / f"eval_syntax_tokens_v04_{report['meta']['label']}.json"
    if not p.is_file():
        return []
    d = json.loads(p.read_text(encoding="utf-8"))
    L = ["", "## v0.4 syntax single-token coverage (plan D4 forced list)", "",
         f"From `{p.relative_to(REPO)}` ({d['num_programs']} canonical programs). "
         "`standalone` = pattern alone encodes to one piece; `unsplit` = in-context occurrences that lie inside one token "
         "(fragment merges count); `exact` = occurrences that are a token by themselves.", "",
         "| model | vocab | standalone single-token | in-context occurrences | unsplit | exact | split occurrences |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for name, mres in d["models"].items():
        sm = mres["summary"]
        L.append(f"| `{name}` | {mres['vocab_size']} | {sm['forced_patterns_single_token_standalone']} "
                 f"({sm['forced_patterns_single_token_standalone_pct']}%) | {sm['forced_occurrences_in_sample']:,} "
                 f"| {sm['forced_unsplit_pct']}% | {sm['forced_exact_pct']}% | {sm['split_occurrences']:,} |")
    zero = [k for k, v in d["pattern_frequency_in_sample"].items() if v == 0]
    if zero:
        L += ["", "Patterns with zero occurrences in the sample (listed in the story but absent from v0.4 corpus text): "
              + ", ".join(f"`{z}`" for z in zero) + "."]
    return L


def wart_section(report: dict[str, Any]) -> list[str]:
    p = REPO / "docs" / f"wart_d4_{report['meta']['label']}.json"
    if not p.is_file():
        return []
    d = json.loads(p.read_text(encoding="utf-8"))
    L = ["", "## D4 AddedToken substring wart (`m= f= t= i=`)", "",
         f"From `{p.relative_to(REPO)}` over {d['num_programs']:,} canonical programs "
         f"({d['total_chars']:,} chars; {len(d.get('min_failures', []))} failed `--min`).", "",
         f"- boundary-aligned `[mfti]=` occurrences (decl heads + `;i=i+1` loop steps): **{d['genuine_decl_heads']:,}**",
         f"- wart occurrences (identifier char + `[mfti]` + `=`, e.g. `result=`, `sum=`, `xi=`): **{d['wart_occurrences']:,}** "
         f"= {d['wart_per_genuine_pct']}% of boundary-aligned; followed by `=` {d['wart_followed_by'].get('=', 0):,} / `==` {d['wart_followed_by'].get('==', 0):,}",
         f"- programs touched: {d['programs_with_wart']:,} ({d['programs_with_wart_pct']}%); {d['warts_per_1k_chars']} per 1k chars",
         "", "| head | boundary-aligned | wart | wart / aligned |", "|---|---:|---:|---:|"]
    for h, v in d["per_head"].items():
        L.append(f"| `{h}` | {v['genuine']:,} | {v['wart']:,} | {v['wart_per_genuine_pct']}% |")
    for r in d.get("realised_on_sp_models", []):
        L.append(f"\nRealised on `{Path(r['model']).relative_to(REPO) if r['model'].startswith(str(REPO)) else r['model']}` "
                 f"(user_defined_symbols): {r['total_mid_identifier']:,} of {r['total_emitted']:,} emitted head pieces "
                 f"({r['mid_identifier_pct']}%) are preceded by an identifier character.")
    L += ["", f"**Recommendation: `{d['recommendation']}`** (material={d['material']}; rule: wart/aligned ≥ "
          f"{d['rule']['material_ratio']:.0%} or programs touched ≥ {d['rule']['material_program_share']:.0%})."]
    return L


def parse_model_arg(s: str) -> tuple[str, str, str]:
    """``name=path`` -> (name, kind, locator); kind inferred from suffix."""
    name, _, loc = s.partition("=")
    if not loc:
        raise argparse.ArgumentTypeError("expected name=path")
    kind = "sp" if loc.endswith(".model") else "hf-json" if loc.endswith(".json") else "hf-hub"
    return name, kind, loc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TEMSpec baseline over the v0.4 sample (131.20)")
    ap.add_argument("--ids", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--label", type=str, required=True, help="pre131 | post131 | ...")
    ap.add_argument("--corpus-note", type=str, default="", help="provenance note for the corpus dir")
    ap.add_argument("--model", type=parse_model_arg, action="append", default=[],
                    help="extra tokenizer name=path (.model | .json | HF id); repeatable")
    ap.add_argument("--only", type=str, default=None, help="comma list of tokenizer names to run")
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--notes", type=Path, default=None, help="markdown appended verbatim to the report")
    ap.add_argument("--out-dir", type=Path, default=REPO)
    ap.add_argument("--tkc", type=Path, default=None)
    args = ap.parse_args(argv)

    tkc = tkcanon.find_tkc(args.tkc)
    print(f"tkc: {tkcanon.tkc_version(tkc)}")
    recs, fails = tkcanon.load_canonical_sample(args.corpus, args.ids, tkc, mask=True)
    print(f"canonical sample: {len(recs)} programs, {len(fails)} failures")
    matches = sum(1 for r in recs if r["file_sha256"] == r["expected_record_sha256"])
    mism = [r["task_id"] for r in recs if r["file_sha256"] != r["expected_record_sha256"]]

    specs = DEFAULT_MODELS + list(args.model)
    if args.only:
        keep = set(args.only.split(","))
        specs = [s for s in specs if s[0] in keep]
    print("loading tokenizers")
    toks, skipped = load_tokenizers(specs, args.allow_download)

    results: dict[str, Any] = {}
    for t in toks:
        print(f"measuring {t.name} ...", flush=True)
        results[t.name] = measure(t, recs)
    add_ratios(results, BASE_TOKENIZER)

    out_data = args.out_dir / "data"
    out_docs = args.out_dir / "docs"
    out_data.mkdir(parents=True, exist_ok=True)
    csv_path = out_data / f"baseline_v04_{args.label}_per_record.csv"
    write_csv(csv_path, recs, results)

    meta = {
        "story": "131.20", "label": args.label, "generated": datetime.now(UTC).isoformat(),
        "tkc_version": tkcanon.tkc_version(tkc), "corpus": str(args.corpus), "corpus_note": args.corpus_note,
        "ids_file": str(args.ids), "ids_file_sha256": hashlib.sha256(args.ids.read_bytes()).hexdigest(),
        "n_records": len(recs), "failures": fails,
        "record_sha_matches": matches, "record_sha_mismatches": mism,
        "sample_min_sha256": hashlib.sha256("\n".join(sorted(r["min_sha256"] for r in recs)).encode()).hexdigest(),
        "sample_masked_sha256": hashlib.sha256("\n".join(r["text"] for r in recs).encode()).hexdigest(),
        "total_chars_masked": sum(len(r["text"]) for r in recs),
        "total_chars_unmasked": sum(len(r["min"]) for r in recs),
        "bootstrap_resamples": BOOTSTRAP_N, "csv_file": str(csv_path.relative_to(args.out_dir)),
        "python": sys.version.split()[0],
    }
    report = {
        "meta": meta, "skipped_tokenizers": skipped, "results": results,
        "records": [{k: r[k] for k in ("task_id", "category", "difficulty", "task_type",
                                       "file_sha256", "min_sha256")} for r in recs],
        "notes_md": args.notes.read_text(encoding="utf-8") if args.notes else "",
    }
    md_path = out_docs / f"baseline_v04_{args.label}.md"
    write_markdown(md_path, report)
    # per-record counts live in the CSV (TEMSpec §5.3); keep the JSON compact.
    slim = {k: {kk: vv for kk, vv in v.items() if kk != "per_record_counts"} for k, v in results.items()}
    json_path = out_data / f"baseline_v04_{args.label}.json"
    json_path.write_text(json.dumps({**report, "results": slim, "notes_md": None}, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {json_path}\n      {csv_path}\n      {md_path}")
    for n, r in results.items():
        mk = r["masked"]
        vs = r.get("vs_cl100k_base", {})
        print(f"  {n:14s} total={mk['total_tokens']:8d} mean={mk['tokens_per_program_mean']:7.1f} "
              f"median={mk['tokens_per_program_median']:6.0f} fert={mk['fertility_mean']:.4f} "
              f"util={100 * r['vocab_utilization']:5.1f}% vs_cl100k={vs.get('ratio_total_tokens', float('nan')):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
