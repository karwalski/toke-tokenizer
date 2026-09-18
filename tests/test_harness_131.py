"""Story 131.20: unit tests for the Phase-0 harness scripts (no corpus / tkc needed)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import baseline_sample
import measure_addedtoken_wart as wart
import tkcanon

# --- baseline_sample ---------------------------------------------------------


def _rows(n_per: dict[tuple[str, int], int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (cat, diff), n in n_per.items():
        for i in range(n):
            rows.append({"task_id": f"{cat}-{i:04d}v1", "category": cat, "difficulty": diff,
                         "sha256": f"{i:064x}"})
    return rows


def test_allocation_is_proportional_and_sums_to_n() -> None:
    sizes = {("A-ARR", 1): 100, ("A-ARR", 2): 300, ("D-WEB", 2): 600}
    alloc = baseline_sample.allocate(sizes, 100)
    assert sum(alloc.values()) == 100
    assert alloc == {("A-ARR", 1): 10, ("A-ARR", 2): 30, ("D-WEB", 2): 60}


def test_allocation_never_exceeds_stratum() -> None:
    alloc = baseline_sample.allocate({("a", 1): 2, ("b", 1): 1000}, 500)
    assert alloc[("a", 1)] <= 2


def test_draw_sample_is_deterministic_and_stratified() -> None:
    rows = _rows({("A-ARR", 1): 50, ("A-ARR", 2): 150, ("D-WEB", 2): 200})
    s1 = baseline_sample.draw_sample(rows, 40, 131)
    s2 = baseline_sample.draw_sample(rows, 40, 131)
    assert [r["task_id"] for r in s1] == [r["task_id"] for r in s2]
    assert len(s1) == 40
    assert sum(1 for r in s1 if r["category"] == "D-WEB") == 20
    assert baseline_sample.draw_sample(rows, 40, 132) != s1


def test_ids_file_roundtrip(tmp_path: Path) -> None:
    rows = _rows({("A-ARR", 1): 5})
    manifest = tmp_path / "MANIFEST.jsonl"
    manifest.write_text("{}\n")
    out = tmp_path / "ids.txt"
    shas = {str(r["task_id"]): "ab" * 32 for r in rows}
    baseline_sample.write_ids(out, rows, manifest, 131, 5, shas, manifest)
    back = tkcanon.read_ids_file(out)
    assert [b["task_id"] for b in back] == [r["task_id"] for r in rows]
    assert back[0]["record_sha256"] == "ab" * 32
    assert back[0]["difficulty"] == "1"


# --- wart counting -------------------------------------------------------------


@pytest.mark.parametrize(
    "text, genuine, warts",
    [
        ("m=a;f=main():i64{<0};", {"m=": 1, "f=": 1}, {}),
        ("m=a;f=g(){let xi=5;<xi};", {"m=": 1, "f=": 1}, {"i=": 1}),
        ("{let result=0;let count=0;let out=mut.0}", {}, {"t=": 3}),
        ("if(cnt==1){<0}", {}, {"t=": 1}),
        ("lp(let i=0;i<n;i=i+1){}", {"i=": 1}, {}),  # ';i=i+1' is boundary-aligned, not a wart
        ("t=point{x:i64};", {"t=": 1}, {}),
        ("let sum=0;let buf=s.builder();", {}, {"m=": 1, "f=": 1}),
    ],
)
def test_wart_counts(text: str, genuine: dict[str, int], warts: dict[str, int]) -> None:
    c = wart.count_text(text)
    assert dict(c["genuine"]) == genuine
    assert dict(c["warts"]) == warts


def test_wart_kinds_distinguish_assignment_from_equality() -> None:
    c = wart.count_text("let cnt=1;if(cnt==2){}")
    assert dict(c["wart_kinds"]) == {"=": 1, "==": 1}


# --- alignment piece normalisation -------------------------------------------


def test_alignment_normalise_piece() -> None:
    pytest.importorskip("transformers")
    import tokenizer_alignment as ta

    assert ta.normalise_piece("▁io", "sentencepiece") == " io"
    assert ta.normalise_piece("Ġio", "bytelevel") == " io"
    assert ta.normalise_piece("Ċ", "bytelevel") == "\n"
    assert ta.normalise_piece("<0x0A>", "sentencepiece") == "\n"
    assert ta.normalise_piece("<unk>", "sentencepiece") is None
    assert ta.normalise_piece("m=", "plain") == "m="
    # a lone continuation byte is not text and must not equal any real string
    assert str(ta.normalise_piece("Ã", "bytelevel")).startswith("<bytes:")
    # the whole point: SP and byte-level spellings of the same token now compare equal
    assert ta.normalise_piece("▁io", "sentencepiece") == ta.normalise_piece("Ġio", "bytelevel")


def test_alignment_hard_fails_without_transformers(tmp_path: Path) -> None:
    """Importing the script with `transformers` unavailable must exit non-zero."""
    import subprocess

    code = (
        "import sys, builtins\n"
        "real = builtins.__import__\n"
        "def fake(name, *a, **k):\n"
        "    if name == 'transformers' or name.startswith('transformers.'):\n"
        "        raise ImportError('blocked')\n"
        "    return real(name, *a, **k)\n"
        "builtins.__import__ = fake\n"
        "sys.argv = ['tokenizer_alignment.py', '--corpus', '/nonexistent']\n"
        "import runpy\n"
        "runpy.run_path(sys.argv[0], run_name='__main__')\n"
    )
    script = Path(__file__).resolve().parent.parent / "scripts" / "tokenizer_alignment.py"
    proc = subprocess.run([sys.executable, "-c", code.replace("tokenizer_alignment.py", str(script), 1)],
                          capture_output=True, text=True, cwd=str(script.parent), check=False)
    assert proc.returncode == 2
    assert "transformers" in proc.stderr
    assert "partial" in proc.stderr.lower()


# --- syntax eval patterns -----------------------------------------------------


def test_syntax_patterns_cover_story_list() -> None:
    import eval_syntax_tokens as est

    surfaces = {s for pats in est.KEY_PATTERNS.values() for s, _ in pats}
    for want in ["==", "!=", "&&", "||", "if(", "el{", "mt ", "\\(", "@(",
                 "$i64", "$f64", "$str", "$bool", "$u64", "$byte", "m=", "f=", "t=", "i=", "lp(", "{<"]:
        assert want in surfaces, want
    assert set(est.FORCED_GROUPS) <= set(est.KEY_PATTERNS)


def test_syntax_in_context_counts_unsplit_vs_exact() -> None:
    import eval_syntax_tokens as est

    class FakeAdapter:
        kind = "fake"

        def spans(self, text: str) -> list[tuple[str, int, int]]:
            # tokens: 'if(' | 'x==' | '1)' -> '==' is inside 'x==' (unsplit, not exact)
            return [("if(", 0, 3), ("x==", 3, 6), ("1)", 6, 8)]

    ic = est.in_context(FakeAdapter(), ["if(x==1)"], {"g": [("if(", r"\bif\("), ("==", r"==")]})  # type: ignore[arg-type]
    rows = {r["pattern"]: r for r in ic["g"]}
    assert rows["if("]["occurrences"] == 1 and rows["if("]["exact"] == 1
    assert rows["=="]["occurrences"] == 1 and rows["=="]["unsplit"] == 1 and rows["=="]["exact"] == 0
