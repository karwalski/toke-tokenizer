#!/usr/bin/env python3
"""Shared canonicalisation helpers for the 131.x tokenizer harness.

* :func:`min_source` / :func:`min_many` -- ``tkc --min`` canonical form (plan D6).
  One process per program (``tkc --min a.tk b.tk`` is unreliable: it hangs and
  emits a single line, measured 2026-09-18), run through a thread pool.
* :func:`mask_strings` -- string-literal masking per plan D2, imported from the
  normative implementation ``toke/scripts/patterns/mask_strings.py`` (story
  131.4).  Located via ``$TOKE_REPO`` (default ``~/tk/toke``).  There is no
  silent fallback: if it cannot be imported the caller gets an ImportError with
  the path that was tried, because the pre/post baselines must mask identically.
* :func:`load_record` -- read one regen_v04 record JSON.
* :func:`encode_via_tokenizers` etc. live in the callers; this module has no
  tokenizer dependencies.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

DEFAULT_TOKE_REPO = Path(os.environ.get("TOKE_REPO", str(Path.home() / "tk" / "toke")))


# ---------------------------------------------------------------------------
# tkc --min
# ---------------------------------------------------------------------------


def find_tkc(explicit: Path | None = None) -> Path:
    """Locate the compiler binary (``$TKC``, then ``$TOKE_REPO/tkc``)."""
    candidates = [explicit, Path(os.environ["TKC"]) if os.environ.get("TKC") else None,
                  DEFAULT_TOKE_REPO / "tkc"]
    for c in candidates:
        if c is not None and c.is_file():
            return c
    raise FileNotFoundError(
        "tkc not found; set $TKC or $TOKE_REPO (tried: "
        + ", ".join(str(c) for c in candidates if c is not None) + ")"
    )


def tkc_version(tkc: Path) -> str:
    out = subprocess.run([str(tkc), "--version"], capture_output=True, text=True, check=False)
    return (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr) else "?"


class MinError(RuntimeError):
    """``tkc --min`` failed for a program."""


def min_source(source: str, tkc: Path, timeout: float = 30.0) -> str:
    """Return the single-line canonical form of ``source`` (no trailing newline)."""
    with tempfile.NamedTemporaryFile("w", suffix=".tk", delete=False, encoding="utf-8") as f:
        f.write(source)
        path = f.name
    try:
        proc = subprocess.run(
            [str(tkc), "--min", path], capture_output=True, text=True, timeout=timeout, check=False
        )
    finally:
        os.unlink(path)
    if proc.returncode != 0 or not proc.stdout.strip():
        raise MinError(f"tkc --min exit {proc.returncode}: {proc.stderr.strip()[:300]}")
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if len(lines) != 1:
        raise MinError(f"tkc --min emitted {len(lines)} lines, expected 1")
    return lines[0]


def min_many(
    sources: Iterable[str], tkc: Path, jobs: int | None = None
) -> list[str | MinError]:
    """``min_source`` over many programs in parallel; failures are returned in place."""
    srcs = list(sources)
    jobs = jobs or max(2, (os.cpu_count() or 4))

    def one(src: str) -> str | MinError:
        try:
            return min_source(src, tkc)
        except MinError as e:
            return e

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(one, srcs))


# ---------------------------------------------------------------------------
# String masking (plan D2) -- normative implementation lives in the toke repo
# ---------------------------------------------------------------------------

_MASK_PATH = DEFAULT_TOKE_REPO / "scripts" / "patterns" / "mask_strings.py"


def load_mask_strings(path: Path = _MASK_PATH) -> Callable[[str], str]:
    """Import ``mask_strings`` from the toke repo's normative module."""
    if not path.is_file():
        raise ImportError(
            f"mask_strings.py not found at {path}; set $TOKE_REPO to a toke checkout "
            "that contains scripts/patterns/mask_strings.py (story 131.4)"
        )
    spec = importlib.util.spec_from_file_location("toke_mask_strings", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("toke_mask_strings", mod)
    spec.loader.exec_module(mod)
    fn: Callable[[str], str] = mod.mask_strings
    return fn


def mask_strings(text: str) -> str:
    """Mask string-literal bodies to ``_`` (keeps ``\\(...)`` interiors)."""
    return load_mask_strings()(text)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def load_record(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        raw = f.read()
    rec: dict[str, Any] = json.loads(raw)
    rec["_file_sha256"] = hashlib.sha256(raw).hexdigest()
    rec["_source_sha256"] = hashlib.sha256(rec["tk_source"].encode("utf-8")).hexdigest()
    return rec


def record_path(corpus: Path, task_id: str, category: str | None = None) -> Path:
    cat = category or task_id[:5]
    return corpus / cat / f"{task_id}.json"


def iter_record_paths(corpus: Path) -> list[Path]:
    """All record JSONs under ``<corpus>/{A,D}-XXX/``, sorted."""
    out: list[Path] = []
    for d in sorted(corpus.iterdir()):
        if d.is_dir() and len(d.name) == 5 and d.name[1] == "-" and d.name[0] in "AD":
            out.extend(sorted(d.glob("*.json")))
    return out


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Canonical sample (ids file from baseline_sample.py + record dir)
# ---------------------------------------------------------------------------


def read_ids_file(path: Path) -> list[dict[str, str]]:
    """Parse ``data/baseline_sample_ids_v04.txt`` (tab-separated, ``#`` comments)."""
    cols = ("task_id", "category", "difficulty", "record_sha256")
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != len(cols):
            raise ValueError(f"malformed ids line in {path}: {line!r}")
        rows.append(dict(zip(cols, parts, strict=True)))
    return rows


def load_canonical_sample(
    corpus: Path,
    ids_file: Path | None,
    tkc: Path,
    mask: bool = True,
    limit: int | None = None,
    jobs: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Load records (all of ``corpus`` or just the ids), ``--min`` them, mask strings.

    Returns ``(records, failures)``.  Each record dict carries ``task_id``,
    ``category``, ``difficulty``, ``file_sha256``, ``source_sha256``,
    ``expected_record_sha256`` (from the ids file, may be ""), ``raw``
    (``tk_source``), ``min`` (canonical), ``text`` (masked canonical when
    ``mask`` else canonical) and ``min_sha256``.
    """
    if ids_file is not None:
        wanted = read_ids_file(ids_file)
        paths = [(record_path(corpus, r["task_id"], r["category"]), r) for r in wanted]
    else:
        paths = [(p, {"task_id": p.stem, "category": p.parent.name, "difficulty": "",
                      "record_sha256": ""}) for p in iter_record_paths(corpus)]
    if limit:
        paths = paths[:limit]

    masker = load_mask_strings() if mask else (lambda s: s)
    recs: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    loaded: list[tuple[dict[str, Any], dict[str, str]]] = []
    for path, meta in paths:
        if not path.is_file():
            failures.append({"task_id": meta["task_id"], "reason": f"missing record {path}"})
            continue
        loaded.append((load_record(path), meta))

    mins = min_many([r["tk_source"] for r, _ in loaded], tkc, jobs=jobs)
    for (rec, meta), m in zip(loaded, mins, strict=True):
        if isinstance(m, MinError):
            failures.append({"task_id": meta["task_id"], "reason": str(m)})
            continue
        regen = rec.get("regen") or {}
        text = masker(m)
        recs.append({
            "task_id": meta["task_id"],
            "category": meta["category"] or regen.get("category") or meta["task_id"][:5],
            "difficulty": int(meta["difficulty"] or regen.get("difficulty") or 0),
            "task_type": regen.get("task_type", ""),
            "file_sha256": rec["_file_sha256"],
            "source_sha256": rec["_source_sha256"],
            "expected_record_sha256": meta.get("record_sha256", ""),
            "raw": rec["tk_source"],
            "min": m,
            "text": text,
            "min_sha256": sha256_text(m),
        })
    return recs, failures
