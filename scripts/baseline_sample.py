#!/usr/bin/env python3
"""Draw the stratified PRE/POST-rewrite baseline sample (story 131.20, plan D7).

Reads a regen_v04 ``MANIFEST.jsonl`` (one record per line with ``task_id``,
``category``, ``difficulty``) and draws ``--n`` records stratified by
category x difficulty (proportional allocation, largest-remainder rounding,
deterministic ``--seed``).  The ids are written to a tab-separated file that
``scripts/baseline_v04.py --ids`` consumes, so the post-rewrite run (131.21)
measures exactly the same records.

The ``record_sha256`` column is the sha256 of the record FILE at freeze time,
taken from ``--record-sha-manifest`` (toke-corpus ``regen/freeze/
freeze_129_manifest.jsonl``).  It is NOT the corpus ``MANIFEST.jsonl`` ``sha256``
field: that one hashes ``tk_source`` and was found stale for 227/2000 sampled
records (not refreshed after the 129.4-5 repair pass), so it cannot serve as
the freeze check.

Usage:
    python3 scripts/baseline_sample.py --manifest /path/to/MANIFEST.jsonl \
        --n 2000 --seed 131 --out data/baseline_sample_ids_v04.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HEADER_COLUMNS = ("task_id", "category", "difficulty", "record_sha256")


def read_manifest(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def allocate(strata_sizes: dict[tuple[str, int], int], n: int) -> dict[tuple[str, int], int]:
    """Proportional allocation with largest-remainder rounding (ties by key)."""
    total = sum(strata_sizes.values())
    if total == 0:
        return {k: 0 for k in strata_sizes}
    n = min(n, total)
    exact = {k: n * size / total for k, size in strata_sizes.items()}
    alloc = {k: int(v) for k, v in exact.items()}
    short = n - sum(alloc.values())
    by_remainder = sorted(strata_sizes, key=lambda k: (-(exact[k] - alloc[k]), k))
    for k in by_remainder[:short]:
        alloc[k] += 1
    # Never allocate more than a stratum holds (only possible after rounding).
    for k, size in strata_sizes.items():
        alloc[k] = min(alloc[k], size)
    return alloc


def draw_sample(rows: list[dict[str, object]], n: int, seed: int) -> list[dict[str, object]]:
    strata: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for r in rows:
        strata[(str(r["category"]), int(str(r["difficulty"])))].append(r)
    alloc = allocate({k: len(v) for k, v in strata.items()}, n)
    rng = random.Random(seed)
    chosen: list[dict[str, object]] = []
    for key in sorted(strata):
        members = sorted(strata[key], key=lambda r: str(r["task_id"]))
        chosen.extend(rng.sample(members, alloc[key]))
    chosen.sort(key=lambda r: str(r["task_id"]))
    return chosen


def write_ids(
    out: Path,
    chosen: list[dict[str, object]],
    manifest: Path,
    seed: int,
    n: int,
    record_shas: dict[str, str] | None = None,
    record_sha_manifest: Path | None = None,
) -> None:
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    lines = [
        "# baseline sample ids (story 131.20, plan D7) -- reuse verbatim for the post-rewrite run",
        f"# manifest={manifest.name} manifest_sha256={manifest_sha}",
        f"# seed={seed} requested_n={n} drawn_n={len(chosen)} strata=category x difficulty",
    ]
    if record_sha_manifest is not None:
        rs = hashlib.sha256(record_sha_manifest.read_bytes()).hexdigest()
        lines.append(
            f"# record_sha256 = sha256 of the record file per {record_sha_manifest.name} "
            f"(sha256={rs}); blank if absent there"
        )
    else:
        lines.append("# record_sha256 = blank (no --record-sha-manifest given)")
    lines.append("# " + "\t".join(HEADER_COLUMNS))
    for r in chosen:
        sha = (record_shas or {}).get(str(r["task_id"]), "")
        lines.append(f"{r['task_id']}\t{r['category']}\t{r['difficulty']}\t{sha}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_ids(path: Path) -> list[dict[str, str]]:
    """Parse an ids file written by :func:`write_ids`."""
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != len(HEADER_COLUMNS):
            raise ValueError(f"malformed ids line: {line!r}")
        rows.append(dict(zip(HEADER_COLUMNS, parts, strict=True)))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=131)
    ap.add_argument("--out", type=Path, default=Path("data/baseline_sample_ids_v04.txt"))
    ap.add_argument(
        "--record-sha-manifest", type=Path, default=None,
        help="freeze manifest JSONL with per-record FILE sha256 (task_id, sha256)",
    )
    args = ap.parse_args(argv)

    rows = read_manifest(args.manifest)
    if not rows:
        print(f"ERROR: no rows in {args.manifest}", file=sys.stderr)
        return 1
    chosen = draw_sample(rows, args.n, args.seed)
    record_shas: dict[str, str] | None = None
    if args.record_sha_manifest is not None:
        record_shas = {
            str(r["task_id"]): str(r["sha256"]) for r in read_manifest(args.record_sha_manifest)
        }
    write_ids(args.out, chosen, args.manifest, args.seed, args.n, record_shas,
              args.record_sha_manifest)
    per_cat: dict[str, int] = defaultdict(int)
    for r in chosen:
        per_cat[str(r["category"])] += 1
    print(f"wrote {len(chosen)} ids to {args.out}")
    for cat in sorted(per_cat):
        print(f"  {cat}: {per_cat[cat]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
