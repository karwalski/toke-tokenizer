"""Tests for the tokenizer artefact provenance check (scripts/check_provenance.py)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import check_provenance as cp  # noqa: E402


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A throwaway git repo with a tracked training script and one artefact."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.invalid")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "train.py").write_text("print('train')\n")
    _git(tmp_path, "add", "train.py")
    _git(tmp_path, "commit", "-q", "-m", "add trainer")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "toke.model").write_bytes(b"\x00model-bytes\x01")
    return tmp_path


def _good_record(root: Path, artefact: Path) -> dict[str, object]:
    return {
        "artifact": artefact.name,
        "sha256": hashlib.sha256((root / artefact).read_bytes()).hexdigest(),
        "training_script": "train.py",
        "training_script_sha": _git(root, "hash-object", "train.py"),
        "config_sha": "a" * 64,
        "corpus_manifest_sha": "b" * 64,
        "created": "2026-09-18",
        "created_by": "test",
    }


def _write(path: Path, record: Mapping[str, object]) -> None:
    path.write_text(json.dumps(record, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def test_passing_case(repo: Path) -> None:
    artefact = Path("models/toke.model")
    _write(repo / "models" / "toke.model.provenance.json", _good_record(repo, artefact))

    report = cp.check(repo)

    assert report.ok, report.violations
    assert report.checked == ["models/toke.model"]
    assert cp.main(["--root", str(repo)]) == 0


def test_missing_provenance_fails(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report = cp.check(repo)

    assert not report.ok
    assert len(report.violations) == 1
    assert "models/toke.model: no provenance record" in report.violations[0]
    assert cp.main(["--root", str(repo)]) == 1
    assert "provenance check FAILED" in capsys.readouterr().err


def test_sha_mismatch_fails(repo: Path) -> None:
    artefact = Path("models/toke.model")
    record = _good_record(repo, artefact)
    record["sha256"] = "0" * 64
    _write(repo / "models" / "toke.model.provenance.json", record)

    report = cp.check(repo)

    assert not report.ok
    assert any("sha256 mismatch" in v for v in report.violations)
    assert cp.main(["--root", str(repo)]) == 1


def test_untracked_training_script_fails(repo: Path) -> None:
    artefact = Path("models/toke.model")
    (repo / "scratch.py").write_text("pass\n")  # exists, but never git-added
    record = _good_record(repo, artefact)
    record["training_script"] = "scratch.py"
    _write(repo / "models" / "toke.model.provenance.json", record)

    report = cp.check(repo)

    assert any("not tracked by git" in v for v in report.violations)


def test_legacy_case(repo: Path) -> None:
    # Root-level tokenizer*.json is in scope too.
    (repo / "tokenizer_v03.json").write_text('{"version": "1.0"}\n')

    written = cp.fix_legacy(repo, created_by="test --fix-legacy")

    assert sorted(p.as_posix() for p in written) == [
        "models/toke.model.provenance.json",
        "tokenizer_v03.provenance.json",
    ]
    record = json.loads((repo / "tokenizer_v03.provenance.json").read_text())
    assert record["legacy"] is True
    assert record["provenance"] == cp.LEGACY_VALUE
    assert record["training_script"] == cp.LEGACY_VALUE
    assert record["sha256"] == hashlib.sha256(b'{"version": "1.0"}\n').hexdigest()

    report = cp.check(repo)
    assert report.ok, report.violations
    assert report.checked == ["models/toke.model", "tokenizer_v03.json"]

    # Legacy still pins bytes: a modified artefact must fail.
    (repo / "tokenizer_v03.json").write_text('{"version": "2.0"}\n')
    assert any("sha256 mismatch" in v for v in cp.check(repo).violations)


def test_unknown_legacy_without_legacy_flag_fails(repo: Path) -> None:
    artefact = Path("models/toke.model")
    record = _good_record(repo, artefact)
    record["corpus_manifest_sha"] = cp.LEGACY_VALUE  # no "legacy": true
    _write(repo / "models" / "toke.model.provenance.json", record)

    report = cp.check(repo)

    assert any('not marked "legacy": true' in v for v in report.violations)


def test_directory_provenance_json_covers_pair(repo: Path) -> None:
    (repo / "models" / "toke.vocab").write_text("vocab\n")
    records = {
        "artifacts": [
            _good_record(repo, Path("models/toke.model")),
            _good_record(repo, Path("models/toke.vocab")),
        ]
    }
    _write(repo / "models" / "provenance.json", records)

    report = cp.check(repo)

    assert report.ok, report.violations
    assert report.checked == ["models/toke.model", "models/toke.vocab"]


def test_orphan_provenance_is_validated_not_fatal(repo: Path) -> None:
    """In CI models/ is gitignored, so records exist without their artefacts."""
    artefact = Path("models/toke.model")
    _write(repo / "models" / "toke.model.provenance.json", _good_record(repo, artefact))
    (repo / "models" / "toke.model").unlink()

    report = cp.check(repo)

    assert report.ok, report.violations
    assert report.checked == []
    assert any("not present; sha256 not verified" in n for n in report.notes)
