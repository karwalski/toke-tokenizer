#!/usr/bin/env python3
"""Tokenizer artefact provenance check (story 131.24).

Every tokenizer artefact — any file under ``models/`` and any ``tokenizer*.json``
at the repo root — must carry a provenance record that names the committed
training script, its blob/commit SHA, the config SHA and the corpus manifest SHA
that produced it.  ``tokenizer_v03.json`` shipped with no committed training
script; ``docs/architecture/tokenizer-v04-plan.md`` (in the ``toke`` repo) says
that failure must not repeat.  This script is the enforcement.

Provenance lookup order for an artefact ``<dir>/<file>``:

1. ``<dir>/<file>.provenance.json``      (e.g. ``toke.model.provenance.json``)
2. ``<dir>/<stem>.provenance.json``      (e.g. ``tokenizer_v03.provenance.json``
   for ``tokenizer_v03.json``)
3. ``<dir>/provenance.json`` — a single record, or ``{"artifacts": [record, ...]}``
   where a record's ``artifact`` names the file.

Required keys: ``artifact``, ``sha256``, ``training_script``,
``training_script_sha``, ``config_sha``, ``corpus_manifest_sha``, ``created``,
``created_by``.  ``sha256`` must always match the file.  The four
training/config/manifest keys may be ``"unknown (legacy)"`` only when the
record also carries ``"legacy": true``.

Usage:
    python scripts/check_provenance.py              # exit 1 on any violation
    python scripts/check_provenance.py --fix-legacy # write legacy records for
                                                    # artefacts that have none
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LEGACY_VALUE = "unknown (legacy)"
PROVENANCE_SUFFIX = ".provenance.json"
DIR_PROVENANCE = "provenance.json"
REQUIRED_KEYS: tuple[str, ...] = (
    "artifact",
    "sha256",
    "training_script",
    "training_script_sha",
    "config_sha",
    "corpus_manifest_sha",
    "created",
    "created_by",
)
# Keys that may be "unknown (legacy)" on a legacy record.
LEGACY_RELAXED_KEYS: tuple[str, ...] = (
    "training_script",
    "training_script_sha",
    "config_sha",
    "corpus_manifest_sha",
)
_HEX_SHA = re.compile(r"^[0-9a-f]{7,64}$")
# Files under models/ that are never artefacts.
_NON_ARTEFACT_NAMES = {".gitkeep", ".gitignore", "README.md", "README", DIR_PROVENANCE}


@dataclass
class Report:
    violations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)

    def fail(self, artefact: str, message: str) -> None:
        self.violations.append(f"{artefact}: {message}")

    def note(self, message: str) -> None:
        self.notes.append(message)

    @property
    def ok(self) -> bool:
        return not self.violations


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def is_provenance_file(path: Path) -> bool:
    return path.name.endswith(PROVENANCE_SUFFIX) or path.name == DIR_PROVENANCE


def find_artefacts(root: Path) -> list[Path]:
    """Return artefact paths (relative to *root*), sorted."""
    found: list[Path] = []
    models = root / "models"
    if models.is_dir():
        for p in sorted(models.rglob("*")):
            if not p.is_file() or is_provenance_file(p):
                continue
            if p.name in _NON_ARTEFACT_NAMES or p.name.startswith("."):
                continue
            found.append(p.relative_to(root))
    for p in sorted(root.glob("tokenizer*.json")):
        if p.is_file() and not is_provenance_file(p):
            found.append(p.relative_to(root))
    return found


def find_provenance_files(root: Path) -> list[Path]:
    """All provenance files in scope (relative to *root*), sorted."""
    found: list[Path] = []
    models = root / "models"
    if models.is_dir():
        found.extend(
            p.relative_to(root)
            for p in sorted(models.rglob("*"))
            if p.is_file() and is_provenance_file(p)
        )
    found.extend(p.relative_to(root) for p in sorted(root.glob(f"*{PROVENANCE_SUFFIX}")))
    return found


def provenance_stem(artefact: Path) -> str:
    """``tokenizer_v03.json`` -> ``tokenizer_v03``; ``toke.model`` -> ``toke.model``."""
    return artefact.stem if artefact.suffix == ".json" else artefact.name


def default_provenance_path(artefact: Path) -> Path:
    """Where ``--fix-legacy`` writes the record for *artefact* (relative path)."""
    return artefact.with_name(provenance_stem(artefact) + PROVENANCE_SUFFIX)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def locate_record(
    root: Path, artefact: Path
) -> tuple[Path | None, dict[str, Any] | None, str | None]:
    """Find the provenance record for *artefact*.

    Returns ``(provenance_path, record, error)``.  ``error`` is set when a
    candidate file exists but is unreadable or malformed.
    """
    candidates = [
        artefact.with_name(artefact.name + PROVENANCE_SUFFIX),
        default_provenance_path(artefact),
    ]
    for rel in candidates:
        path = root / rel
        if path.is_file():
            try:
                data = _load_json(path)
            except (OSError, ValueError) as exc:
                return rel, None, f"unreadable provenance file {rel}: {exc}"
            if not isinstance(data, dict):
                return rel, None, f"provenance file {rel} must contain a JSON object"
            return rel, data, None

    dir_rel = artefact.parent / DIR_PROVENANCE
    dir_path = root / dir_rel
    if dir_path.is_file():
        try:
            data = _load_json(dir_path)
        except (OSError, ValueError) as exc:
            return dir_rel, None, f"unreadable provenance file {dir_rel}: {exc}"
        records: list[Any]
        if isinstance(data, dict) and isinstance(data.get("artifacts"), list):
            records = data["artifacts"]
        elif isinstance(data, dict):
            records = [data]
        else:
            return dir_rel, None, f"provenance file {dir_rel} must contain a JSON object"
        for rec in records:
            names = (artefact.name, artefact.as_posix())
            if isinstance(rec, dict) and rec.get("artifact") in names:
                return dir_rel, rec, None
    return None, None, None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_tracked_files(root: Path) -> set[str] | None:
    """Repo-relative paths tracked by git, or ``None`` if *root* is not a repo."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return {p.decode("utf-8") for p in out.split(b"\0") if p}


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def validate_record(
    record: dict[str, Any],
    artefact: Path,
    root: Path,
    tracked: set[str] | None,
    report: Report,
    *,
    check_sha: bool = True,
) -> None:
    """Append violations for *record* (describing *artefact*) to *report*."""
    label = artefact.as_posix()
    missing = [k for k in REQUIRED_KEYS if k not in record]
    if missing:
        report.fail(label, f"provenance missing required keys: {', '.join(missing)}")

    legacy = record.get("legacy") is True
    if legacy and record.get("provenance") != LEGACY_VALUE:
        report.fail(label, f'legacy record must carry "provenance": "{LEGACY_VALUE}"')

    artifact_field = record.get("artifact")
    if artifact_field not in (artefact.name, label):
        report.fail(
            label, f"provenance 'artifact' is {artifact_field!r}, expected {artefact.name!r}"
        )

    recorded_sha = record.get("sha256")
    if not _nonempty_str(recorded_sha):
        report.fail(label, "provenance 'sha256' missing or empty")
    elif check_sha:
        actual = sha256_of(root / artefact)
        if actual != recorded_sha:
            report.fail(label, f"sha256 mismatch: provenance says {recorded_sha}, file is {actual}")

    for key in LEGACY_RELAXED_KEYS:
        value = record.get(key)
        if value == LEGACY_VALUE:
            if not legacy:
                report.fail(
                    label,
                    f"'{key}' is \"{LEGACY_VALUE}\" but record is not marked \"legacy\": true",
                )
            continue
        if not isinstance(value, str) or not value.strip():
            if key in record:
                report.fail(label, f"'{key}' must be a non-empty string")
            continue
        if key == "training_script":
            script_rel = Path(value)
            if script_rel.is_absolute() or not (root / script_rel).is_file():
                report.fail(
                    label, f"training_script {value!r} does not exist (must be repo-relative)"
                )
            elif tracked is not None and script_rel.as_posix() not in tracked:
                report.fail(label, f"training_script {value!r} is not tracked by git")
        elif key == "training_script_sha" and not _HEX_SHA.match(value):
            report.fail(label, f"training_script_sha {value!r} is not a git blob/commit sha")

    for key in ("created", "created_by"):
        if key in record and not _nonempty_str(record.get(key)):
            report.fail(label, f"'{key}' must be a non-empty string")


def check(root: Path) -> Report:
    """Run the full check over *root* and return a :class:`Report`."""
    report = Report()
    tracked = git_tracked_files(root)
    if tracked is None:
        report.note("not a git checkout: training_script tracking not verified")

    artefacts = find_artefacts(root)
    covered: set[Path] = set()
    for artefact in artefacts:
        prov_path, record, error = locate_record(root, artefact)
        if error is not None:
            report.fail(artefact.as_posix(), error)
            continue
        if record is None:
            report.fail(
                artefact.as_posix(),
                f"no provenance record (expected {default_provenance_path(artefact).as_posix()} "
                f"or {(artefact.parent / DIR_PROVENANCE).as_posix()})",
            )
            continue
        assert prov_path is not None
        covered.add(prov_path)
        report.checked.append(artefact.as_posix())
        validate_record(record, artefact, root, tracked, report)

    # Provenance files whose artefact is absent (models/ is gitignored, so in CI
    # this is the normal case).  Validate the record's shape; skip the sha check.
    for prov in find_provenance_files(root):
        if prov in covered:
            continue
        try:
            data = _load_json(root / prov)
        except (OSError, ValueError) as exc:
            report.fail(prov.as_posix(), f"unreadable provenance file: {exc}")
            continue
        records: list[Any]
        if isinstance(data, dict) and isinstance(data.get("artifacts"), list):
            records = data["artifacts"]
        elif isinstance(data, dict):
            records = [data]
        else:
            report.fail(prov.as_posix(), "provenance file must contain a JSON object")
            continue
        for rec in records:
            if not isinstance(rec, dict):
                report.fail(prov.as_posix(), "each provenance record must be a JSON object")
                continue
            name = rec.get("artifact")
            if not isinstance(name, str) or not name.strip():
                report.fail(prov.as_posix(), "provenance 'artifact' missing or empty")
                continue
            artefact = prov.parent / Path(name).name
            if (root / artefact).is_file():
                continue  # covered above via a different lookup path
            report.note(
                f"{prov.as_posix()}: artefact {artefact.as_posix()} not present; "
                "sha256 not verified"
            )
            validate_record(rec, artefact, root, tracked, report, check_sha=False)
    return report


# ---------------------------------------------------------------------------
# --fix-legacy
# ---------------------------------------------------------------------------


def legacy_record(root: Path, artefact: Path, created_by: str) -> dict[str, Any]:
    return {
        "artifact": artefact.name,
        "sha256": sha256_of(root / artefact),
        "training_script": LEGACY_VALUE,
        "training_script_sha": LEGACY_VALUE,
        "config_sha": LEGACY_VALUE,
        "corpus_manifest_sha": LEGACY_VALUE,
        "created": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d"),
        "created_by": created_by,
        "legacy": True,
        "provenance": LEGACY_VALUE,
        "note": (
            "Artefact predates the provenance rule (story 131.24); its generator "
            "and inputs were not recorded. Do not treat as reproducible."
        ),
    }


def fix_legacy(root: Path, created_by: str) -> list[Path]:
    """Write legacy records for artefacts that have none. Returns paths written."""
    written: list[Path] = []
    for artefact in find_artefacts(root):
        _path, record, error = locate_record(root, artefact)
        if record is not None or error is not None:
            continue
        target = root / default_provenance_path(artefact)
        with target.open("w", encoding="utf-8") as fh:
            json.dump(legacy_record(root, artefact, created_by), fh, indent=2)
            fh.write("\n")
        written.append(target.relative_to(root))
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repository root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--fix-legacy",
        action="store_true",
        help="write a legacy provenance record for every artefact that has none, then check",
    )
    parser.add_argument(
        "--created-by",
        default="scripts/check_provenance.py --fix-legacy",
        help="value for 'created_by' in records written by --fix-legacy",
    )
    args = parser.parse_args(argv)
    root: Path = args.root.resolve()

    if args.fix_legacy:
        for rel in fix_legacy(root, args.created_by):
            print(f"wrote legacy provenance: {rel.as_posix()}")

    report = check(root)
    for note in report.notes:
        print(f"note: {note}")
    for artefact in report.checked:
        print(f"ok: {artefact}")
    if report.ok:
        print(f"provenance check passed ({len(report.checked)} artefact(s) verified)")
        return 0
    print(f"provenance check FAILED: {len(report.violations)} violation(s)", file=sys.stderr)
    for v in report.violations:
        print(f"  - {v}", file=sys.stderr)
    print(
        "Every file under models/ and every tokenizer*.json at the repo root needs a "
        "<name>.provenance.json (or provenance.json in its directory). See CONTRIBUTING.md.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
