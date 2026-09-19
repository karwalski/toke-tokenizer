#!/usr/bin/env python3
"""Derive the v0.4 tokenizer must-merge list from the pattern catalogue.

Story 131.23 (Epic 131); consumes ``toke/patterns/catalogue*.json`` (story
131.7, wave 2 arrives as ``catalogue.wave2.json``) and writes
``data/must_merge_v04.json`` -- the list 116.9 Phase 2 seeds from and Phase 3
gates on (plan ``docs/architecture/tokenizer-v04-plan.md`` §3 D4, as amended by
131.20).  READ-ONLY on the catalogue, the fixtures, the corpus and the compiler.

Pipeline
--------
1. For every catalogue entry take the **canonical** form and, when set, the
   **hot_path** form (``verdict.canonical`` / ``verdict.hot_path``).  The
   fixture ``patterns/<id>/<form>.tk`` is ``tkc --min``'d, the ``f=pat`` function
   is isolated (``count_tokens.function_extent``, story 131.4) and its string
   bodies are masked to ``_`` (``mask_strings.py``, plan D2).  A fixture that
   cannot be minified / has no ``pat`` / only exists as ``<form>.blocked.tk`` is
   reported under ``unparsed`` and skipped -- never silently.
2. The masked ``pat`` text is split into **syntax fragments** (grammar below).
3. Each fragment is counted over the canonical forms (extraction identity) and
   over a corpus sample (substring occurrences, identifier-boundary guarded --
   the frequency a BPE merge actually sees).  The sample is ``--sample N``
   accepted ``regen_v04`` records drawn with ``random.Random(--seed)`` (131),
   ``tkc --min``'d in parallel and masked; it is written to ``--sample-file`` so
   the derivation is reproducible without the corpus (the pytest derivability
   test re-derives from the committed sample file and the committed catalogue
   and asserts byte identity with the committed data file).
4. Classification (D4 as amended by 131.20):
   * ``forced``  -- the closed-class list ``FORCED_D4``: operators
     ``== != && || <= >=``, ``@(``, ``\\(``, ``lp(``, ``mt ``, ``if(``, ``el{``
     and the declaration heads ``m= f= t= i=`` (**seeded merges, not
     AddedTokens** -- 131.20 measured the AddedToken substring wart at 17.1% of
     heads / 39.1% of programs); plus any fragment present in >= 50% of the
     canonical forms with corpus programs share >= 5% (``PROMOTE_*``).
   * ``verify``  -- everything else extracted from a canonical/hot-path form:
     must merge naturally; the Phase 3 gate checks it (unsplit in context).
   * ``excluded`` (separate key, never in ``fragments``): the D4 type sigils
     ``$i64 $f64 $str $bool $u64 $byte`` -- 131.20 found ZERO occurrences in
     v0.4 text (types are ``:i64`` / ``@i64``; ``$`` only opens ``$ok:`` /
     ``$err:`` arms and user type names).  Their corpus count is still reported
     so the exclusion stays evidence-based.  ``$ok:`` / ``$err:`` are ordinary
     head fragments and are classified by the rules above.
   Single-character fragments are dropped: every byte is a base token of a
   byte-level BPE, so "must merge" is vacuous for them.

Fragment grammar (``fragments()``)
---------------------------------
Input: one ``tkc --min`` program (or function) with string bodies masked.
Characters are classed as

    IDENT  ``[a-z][a-z0-9]*``    (v0.4 has no uppercase and no ``_`` in code)
    NUM    ``[0-9]+(\\.[0-9]+)?``
    MASK   ``_``                  (a masked string body)
    PUNCT  anything else -- operators, brackets, ``;`` ``:`` ``"`` ``\\`` ``$``
           ``@`` ``&`` ``!`` and the space.

IDENT, NUM and MASK are *atoms*: they are never part of a fragment and they end
the fragment before them.  A fragment is one of

  R  a maximal run of PUNCT characters, e.g. ``){<`` ``};`` ``==`` ``)")``;
     a TYPE name (``i64 f64 str bool u64 byte``) directly preceded by ``:`` or
     ``@`` (type position) is absorbed into the run together with the PUNCT that
     follows it, e.g. ``:i64):i64{`` ``:@i64;`` ``:str{<`` -- plan D4 calls these
     fragment merges a feature.
  H  a *head*: a closed-class keyword plus the single opening character that
     always follows it in ``--min`` text -- ``el if(`` ``if(`` ``el{`` ``lp(``
     ``mt `` ``let `` ``mut.`` -- or a result-arm head ``$ok:`` ``$err:``.  A head
     ends the run before it and is itself one fragment (so ``}el{<`` splits into
     ``}`` ``el{`` ``<``; counts aggregate per head instead of per context).
  M  a *member head*: ``.`` + IDENT (+ ``(`` when the member is called), e.g.
     ``.get(`` ``.len`` ``.println(`` ``.io`` -- open-class stdlib surface, always
     ``verify`` unless promoted.  Like H it stands alone (``).get(`` is ``)`` +
     ``.get(``).
  D  a *declaration head*: ``m= f= t= i=`` at brace depth 0 -- inside a function
     ``i=i+1`` is IDENT ``i`` + run ``=`` (the substring corpus count for ``i=``
     etc. is nevertheless boundary-guarded, not depth-guarded, because a seeded
     merge applies wherever the bigram occurs -- the 131.20 wart measurement).

Corpus counting: ``count_occurrences(fragment, text)`` counts non-overlapping
substring matches of the literal fragment with ``(?<![a-z0-9])`` / ``(?![a-z0-9])``
guards on an alphanumeric first / last character (so ``lp(`` does not match
``help(`` and ``.len`` does not match ``.length``).

Usage
-----
    python3 scripts/derive_must_merge.py \\
        --catalogue ~/tk/toke/patterns/catalogue.json [--catalogue ...wave2.json] \\
        --corpus ~/tk/toke-corpus/corpus/regen_v04 --sample 5000 --seed 131 \\
        --sample-file data/must_merge_sample_v04.txt --out data/must_merge_v04.json

Without ``--corpus`` the sample is READ from ``--sample-file`` (hermetic re-run).
Without ``--catalogue`` every ``$TOKE_REPO/patterns/catalogue*.json`` is used.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tkcanon

REPO = Path(__file__).resolve().parent.parent
GRAMMAR_VERSION = 1
FUNCTION = "pat"

# D4 closed-class forced list, as amended by 131.20 (sigils out, heads = seeded merges).
FORCED_D4: dict[str, str] = {
    "==": "D4 operator", "!=": "D4 operator", "&&": "D4 operator", "||": "D4 operator",
    "<=": "D4 operator", ">=": "D4 operator",
    "@(": "D4 collection open", "\\(": "D4 interpolation open",
    "lp(": "D4 loop head", "mt ": "D4 match head (surface form in --min text is `mt `)",
    "if(": "D4 (131.20 amendment) expression/statement-if head",
    "el{": "D4 (131.20 amendment) else head",
    "m=": "D4 declaration head -- SEEDED MERGE, not AddedToken (131.20 wart 17.1%)",
    "f=": "D4 declaration head -- SEEDED MERGE, not AddedToken (131.20 wart 17.1%)",
    "t=": "D4 declaration head -- SEEDED MERGE, not AddedToken (131.20 wart: `t=` dominates)",
    "i=": "D4 declaration head -- SEEDED MERGE, not AddedToken (131.20 wart 17.1%)",
}
EXCLUDED_SIGILS = ("$i64", "$f64", "$str", "$bool", "$u64", "$byte")
EXCLUDED_NOTE = (
    "D4 type sigil EXCLUDED from forced: 131.20 found zero occurrences in v0.4 text "
    "(types are `:i64` / `@i64`; `$` only opens `$ok:`/`$err:` arms and user type names)"
)
PROMOTE_CANONICAL_SHARE = 0.5
PROMOTE_CORPUS_SHARE = 0.05

TYPES = ("i64", "f64", "str", "bool", "u64", "byte")
HEADS = ("el if(", "if(", "el{", "lp(", "mt ", "let ", "mut.", "$ok:", "$err:")  # longest first
DECL_HEADS = ("m=", "f=", "t=", "i=")

_IDENT_START = set("abcdefghijklmnopqrstuvwxyz")
_IDENT_CHAR = _IDENT_START | set("0123456789")
_DIGIT = set("0123456789")


# ---------------------------------------------------------------------------
# Fragment grammar
# ---------------------------------------------------------------------------


def _ident_end(text: str, i: int) -> int:
    n = len(text)
    while i < n and text[i] in _IDENT_CHAR:
        i += 1
    return i


def fragments(text: str) -> list[str]:
    """Split masked ``--min`` text into syntax fragments (grammar in the module doc)."""
    out: list[str] = []
    run: list[str] = []
    n = len(text)
    depth = 0
    i = 0

    def flush() -> None:
        if run:
            out.append("".join(run))
            run.clear()

    while i < n:
        c = text[i]
        prev = text[i - 1] if i > 0 else ""
        # --- heads (only at an identifier boundary) ------------------------
        if (c in _IDENT_START or c == "$") and prev not in _IDENT_CHAR:
            head = next((h for h in HEADS if text.startswith(h, i)), None)
            if head is not None:
                flush()
                out.append(head)
                i += len(head)
                continue
        if c in _IDENT_START:
            j = _ident_end(text, i)
            word = text[i:j]
            if depth == 0 and word + "=" in DECL_HEADS and text.startswith("=", j) \
                    and prev not in _IDENT_CHAR:
                flush()
                out.append(word + "=")
                i = j + 1
                continue
            if word in TYPES and prev in ":@":
                run.append(word)  # type position: absorbed into the run
                i = j
                continue
            if prev == "." and run and run[-1] == "." and not (len(run) >= 2 and run[-2] in _DIGIT):
                run.pop()  # member head: `.name` or `.name(`
                flush()
                if text.startswith("(", j):
                    out.append("." + word + "(")
                    i = j + 1
                else:
                    out.append("." + word)
                    i = j
                continue
            flush()  # IDENT atom
            i = j
            continue
        if c in _DIGIT:
            j = i + 1
            while j < n and text[j] in _DIGIT:
                j += 1
            if j + 1 < n and text[j] == "." and text[j + 1] in _DIGIT:
                j += 2
                while j < n and text[j] in _DIGIT:
                    j += 1
            flush()  # NUM atom
            i = j
            continue
        if c == "_":
            flush()  # MASK atom
            i += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        run.append(c)
        i += 1
    flush()
    return out


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


def _guarded(fragment: str) -> re.Pattern[str]:
    pat = re.escape(fragment)
    if fragment[0] in _IDENT_CHAR:
        pat = "(?<![a-z0-9])" + pat
    if fragment[-1] in _IDENT_CHAR:
        pat = pat + "(?![a-z0-9])"
    return re.compile(pat)


def count_occurrences(fragment: str, text: str) -> int:
    """Non-overlapping, identifier-boundary-guarded substring occurrences."""
    return sum(1 for _ in _guarded(fragment).finditer(text))


def corpus_counts(frags: list[str], texts: list[str]) -> dict[str, tuple[int, int]]:
    """fragment -> (occurrences, programs containing it) over ``texts``."""
    out: dict[str, tuple[int, int]] = {}
    for f in frags:
        rx = _guarded(f)
        occ = progs = 0
        for t in texts:
            k = sum(1 for _ in rx.finditer(t))
            if k:
                occ += k
                progs += 1
        out[f] = (occ, progs)
    return out


# ---------------------------------------------------------------------------
# Catalogue + fixtures
# ---------------------------------------------------------------------------


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_catalogues(paths: list[Path]) -> list[dict[str, Any]]:
    out = []
    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        out.append({"path": p, "sha256": sha256_file(p), "protocol": data.get("protocol"),
                    "entries": data["entries"]})
    return out


def catalogue_forms(cats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every (entry, role, form, fixture) to extract -- canonical and hot_path."""
    forms = []
    for cat in cats:
        for e in cat["entries"]:
            v = e.get("verdict") or {}
            for role in ("canonical", "hot_path"):
                form = v.get(role)
                if not form:
                    continue
                cand = next((c for c in e.get("candidates", []) if c.get("form") == form), None)
                fixture = (cand or {}).get("fixture") or f"patterns/{e['id']}/{form}.tk"
                forms.append({"id": e["id"], "role": role, "form": form, "fixture": fixture,
                              "status": v.get("status"), "catalogue": cat["path"].name})
    return forms


def load_count_tokens(toke_repo: Path) -> Any:
    p = toke_repo / "scripts" / "patterns" / "count_tokens.py"
    if not p.is_file():
        raise ImportError(f"count_tokens.py not found at {p}; set $TOKE_REPO")
    spec = importlib.util.spec_from_file_location("toke_count_tokens", p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("toke_count_tokens", mod)
    spec.loader.exec_module(mod)
    return mod


def make_pat_extractor(toke_repo: Path, tkc: Path) -> Callable[[Path], str]:
    """Return ``fixture path -> masked --min text of f=pat`` backed by tkc."""
    ct = load_count_tokens(toke_repo)
    os.environ.setdefault("TKC", str(tkc))
    ct.TKC = tkc
    mask = tkcanon.load_mask_strings(toke_repo / "scripts" / "patterns" / "mask_strings.py")

    def extract(fixture: Path) -> str:
        m = ct.min_text(fixture)
        s, e = ct.function_extent(m, FUNCTION)
        return str(mask(m[s:e]))

    return extract


# ---------------------------------------------------------------------------
# Corpus sample file
# ---------------------------------------------------------------------------


def accepted_record_paths(corpus: Path) -> list[Path]:
    out = []
    for p in tkcanon.iter_record_paths(corpus):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if (rec.get("judge") or {}).get("accepted") and \
                (rec.get("validation") or {}).get("compiler_exit_code") == 0:
            out.append(p)
    return out


def draw_sample(corpus: Path, n: int, seed: int, tkc: Path, jobs: int | None = None
                ) -> tuple[dict[str, Any], list[tuple[str, str]], list[dict[str, str]]]:
    """(meta, [(task_id, masked_min)], failures) for ``n`` seeded accepted records."""
    paths = accepted_record_paths(corpus)
    picked = sorted(random.Random(seed).sample(paths, min(n, len(paths))), key=lambda p: p.stem)
    srcs = [json.loads(p.read_text(encoding="utf-8"))["tk_source"] for p in picked]
    mins = tkcanon.min_many(srcs, tkc, jobs=jobs)
    mask = tkcanon.load_mask_strings()
    rows: list[tuple[str, str]] = []
    failures: list[dict[str, str]] = []
    for p, m in zip(picked, mins, strict=True):
        if isinstance(m, tkcanon.MinError):
            failures.append({"task_id": p.stem, "reason": str(m)})
        else:
            rows.append((p.stem, mask(m)))
    shown = str(corpus.resolve())
    home = str(Path.home())
    if shown.startswith(home):
        shown = "~" + shown[len(home):]
    meta = {"corpus": shown, "accepted_records": len(paths), "requested": n, "seed": seed,
            "tkc_version": tkcanon.tkc_version(tkc), "min_failures": len(failures)}
    return meta, rows, failures


def write_sample(path: Path, meta: dict[str, Any], rows: list[tuple[str, str]]) -> None:
    lines = [f"# {k}: {v}" for k, v in meta.items()]
    lines.append("# format: task_id<TAB>masked tkc --min text (plan D2/D6), one record per line")
    lines.extend(f"{tid}\t{text}" for tid, text in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_sample(path: Path) -> tuple[dict[str, str], list[tuple[str, str]]]:
    meta: dict[str, str] = {}
    rows: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            if ":" in line and not line.startswith("# format"):
                k, v = line[2:].split(":", 1)
                meta[k.strip()] = v.strip()
            continue
        if not line.strip():
            continue
        tid, text = line.split("\t", 1)
        rows.append((tid, text))
    return meta, rows


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------


def derive(cats: list[dict[str, Any]], extract: Callable[[Path], str], toke_repo: Path,
           sample_meta: dict[str, Any], sample_rows: list[tuple[str, str]],
           sample_sha256: str) -> dict[str, Any]:
    forms = catalogue_forms(cats)
    per_form: list[dict[str, Any]] = []
    unparsed: list[dict[str, str]] = []
    for f in forms:
        fixture = toke_repo / f["fixture"]
        if not fixture.is_file():
            alt = fixture.with_name(f"{f['form']}.blocked.tk")
            unparsed.append({**{k: str(v) for k, v in f.items()},
                             "reason": f"fixture missing ({'blocked: ' + alt.name if alt.is_file() else 'no file'})"})
            continue
        try:
            text = extract(fixture)
        # BLE001 suppressed: `extract` shells out to tkc and parses its output, so the failure modes
        # are open-ended (CalledProcessError, ValueError, IndexError, UnicodeDecodeError).
        # Every one of them must land in `unparsed` with its type recorded, never abort the run.
        except Exception as e:  # noqa: BLE001 - tkc failure, no `pat`, unbalanced braces ...
            unparsed.append({**{k: str(v) for k, v in f.items()},
                             "reason": f"{type(e).__name__}: {str(e)[:200]}"})
            continue
        fr = fragments(text)
        counts: dict[str, int] = {}
        for x in fr:
            counts[x] = counts.get(x, 0) + 1
        per_form.append({**f, "masked_pat": text, "fragments": counts})

    canonical = [p for p in per_form if p["role"] == "canonical"]
    n_canonical = len(canonical)
    all_frags: set[str] = set()
    for p in per_form:
        all_frags.update(k for k in p["fragments"] if len(k) >= 2)
    all_frags.update(FORCED_D4)
    texts = [t for _, t in sample_rows]
    cc = corpus_counts(sorted(all_frags) + list(EXCLUDED_SIGILS), texts)
    n_prog = len(texts)

    rows = []
    for frag in sorted(all_frags):
        in_can = sorted({p["id"] for p in canonical if frag in p["fragments"]})
        in_hot = sorted({p["id"] for p in per_form if p["role"] == "hot_path" and frag in p["fragments"]})
        can_occ = sum(p["fragments"].get(frag, 0) for p in canonical)
        occ, progs = cc[frag]
        share = round(progs / n_prog, 4) if n_prog else 0.0
        can_share = len(in_can) / n_canonical if n_canonical else 0.0
        if frag in FORCED_D4:
            cls, why = "forced", FORCED_D4[frag]
            if not in_can:
                why += "; not in any current canonical form (closed-class, listed regardless)"
        elif can_share >= PROMOTE_CANONICAL_SHARE and share >= PROMOTE_CORPUS_SHARE:
            cls = "forced"
            why = (f"promoted: in {len(in_can)}/{n_canonical} canonical forms "
                   f"(>= {PROMOTE_CANONICAL_SHARE:.0%}) and {share:.1%} of corpus programs "
                   f"(>= {PROMOTE_CORPUS_SHARE:.0%})")
        else:
            cls = "verify"
            why = (f"open-class / contextual fragment: in {len(in_can)}/{n_canonical} canonical forms, "
                   f"{share:.1%} of corpus programs -- must merge naturally (Phase 3 gate)")
        rows.append({"fragment": frag, "in_canonical_forms": in_can, "in_hot_path_forms": in_hot,
                     "canonical_occurrences": can_occ, "corpus_freq": occ, "corpus_programs": progs,
                     "corpus_programs_share": share, "class": cls, "rationale": why})
    rows.sort(key=lambda r: (r["class"] != "forced", -r["corpus_freq"], r["fragment"]))

    excluded = []
    for s in EXCLUDED_SIGILS:
        occ, progs = cc[s]
        excluded.append({"fragment": s, "corpus_freq": occ, "corpus_programs": progs,
                         "class": "excluded", "rationale": EXCLUDED_NOTE})

    return {
        "story": "131.23",
        "generated_by": "scripts/derive_must_merge.py",
        "grammar_version": GRAMMAR_VERSION,
        "function": FUNCTION,
        "catalogues": [{"path": str(c["path"].name), "sha256": c["sha256"], "protocol": c["protocol"],
                        "entries": len(c["entries"])} for c in cats],
        "corpus_sample": {**sample_meta, "records": n_prog, "sample_file_sha256": sample_sha256},
        "thresholds": {"promote_canonical_share": PROMOTE_CANONICAL_SHARE,
                       "promote_corpus_programs_share": PROMOTE_CORPUS_SHARE,
                       "min_fragment_length": 2},
        "summary": {
            "forms_extracted": len(per_form), "canonical_forms": n_canonical,
            "hot_path_forms": len(per_form) - n_canonical, "unparsed": len(unparsed),
            "forced": sum(1 for r in rows if r["class"] == "forced"),
            "verify": sum(1 for r in rows if r["class"] == "verify"),
        },
        "forms": [{k: v for k, v in p.items() if k != "fragments"} | {"fragments": p["fragments"]}
                  for p in per_form],
        "unparsed": unparsed,
        "fragments": rows,
        "excluded": excluded,
    }


def render(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=2, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def default_catalogues(toke_repo: Path) -> list[Path]:
    return sorted((toke_repo / "patterns").glob("catalogue*.json"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalogue", action="append", type=Path,
                    help="catalogue JSON (repeatable; default: $TOKE_REPO/patterns/catalogue*.json)")
    ap.add_argument("--toke-repo", type=Path, default=tkcanon.DEFAULT_TOKE_REPO)
    ap.add_argument("--tkc", type=Path, default=None)
    ap.add_argument("--corpus", type=Path, help="regen_v04 root: draw a fresh sample and write --sample-file")
    ap.add_argument("--sample", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=131)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--sample-file", type=Path, default=REPO / "data" / "must_merge_sample_v04.txt")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "must_merge_v04.json")
    a = ap.parse_args(argv)

    tkc = tkcanon.find_tkc(a.tkc)
    cat_paths = a.catalogue or default_catalogues(a.toke_repo)
    if not cat_paths:
        ap.error("no catalogue files found")
    cats = load_catalogues(cat_paths)

    if a.corpus:
        meta, rows, failures = draw_sample(a.corpus, a.sample, a.seed, tkc, a.jobs)
        a.sample_file.parent.mkdir(parents=True, exist_ok=True)
        write_sample(a.sample_file, meta, rows)
        for f in failures:
            print(f"min failure: {f['task_id']}: {f['reason']}", file=sys.stderr)
        print(f"sample: {len(rows)} records -> {a.sample_file} ({len(failures)} --min failures)",
              file=sys.stderr)
    smeta, rows = read_sample(a.sample_file)
    result = derive(cats, make_pat_extractor(a.toke_repo, tkc), a.toke_repo, dict(smeta), rows,
                    sha256_file(a.sample_file))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(render(result), encoding="utf-8")
    s = result["summary"]
    print(f"{a.out}: {s['forced']} forced / {s['verify']} verify from {s['canonical_forms']} canonical "
          f"+ {s['hot_path_forms']} hot-path forms; {s['unparsed']} unparsed", file=sys.stderr)
    for u in result["unparsed"]:
        print(f"  unparsed {u['id']}/{u['form']} ({u['role']}): {u['reason']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
