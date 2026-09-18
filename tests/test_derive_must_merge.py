"""Story 131.23: must-merge list derived from the pattern catalogue.

* grammar / counting unit tests (pure, no tkc);
* a synthetic catalogue + stub fixture extractor -> expected fragments and
  classes, with the corpus part fed from the small hermetic sample under
  ``tests/fixtures/`` (no corpus, no tkc);
* the **derivability test** (the story's acceptance): the committed
  ``data/must_merge_v04.json`` must be byte-identical to a fresh derivation from
  the committed catalogue(s) in the toke repo and the committed corpus sample
  ``data/must_merge_sample_v04.txt`` -- no hand edits survive.  Needs ``tkc`` and
  the toke checkout (skipped otherwise, like the other 131.x live tests).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import derive_must_merge as dmm
import tkcanon

REPO = Path(__file__).resolve().parent.parent
SMALL_SAMPLE = REPO / "tests" / "fixtures" / "must_merge_sample_small.txt"
DATA = REPO / "data" / "must_merge_v04.json"
SAMPLE = REPO / "data" / "must_merge_sample_v04.txt"

# --- grammar --------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("f=pat(x:i64):i64{if(x>0){<1*x+1}el{<2*x+2}}",
         ["f=", "(", ":i64):i64{", "if(", ">", "){<", "*", "+", "}", "el{", "<", "*", "+", "}}"]),
        ("f=pat(xs:@i64):i64{<xs.sort(&cmp).get(0)}",
         ["f=", "(", ":@i64):i64{<", ".sort(", "&", ")", ".get(", ")}"]),
        ("f=pat(x:i64):i64{<mt chk(x){$ok:v v;$err:e-1}}",
         ["f=", "(", ":i64):i64{<", "mt ", "(", "){", "$ok:", " ", ";", "$err:", "-", "}}"]),
        ("f=pat(x:i64):i64{<if(x>90){4}el if(x>80){3}el{1}}",
         ["f=", "(", ":i64):i64{<", "if(", ">", "){", "}", "el if(", ">", "){", "}", "el{", "}}"]),
        # decl heads only at depth 0; `i=i+1` inside a function is IDENT + `=`; `.5` is a number
        ("m=main;i=io:std.io;f=main():i64{lp(let i=0;i<n;i=i+1){x=0.5};<0};",
         ["m=", ";", "i=", ":", ".io", ";", "f=", "():i64{", "lp(", "let ", "=", ";", "<", ";", "=",
          "+", "){", "=", "};<", "};"]),
        # masked string body `_` is an atom; `\(` interpolation and quotes are punctuation
        ('io.println("_\\(x) \\(y.len)")', [".println(", '"', "\\(", ") \\(", ".len", ')")']),
        # keyword heads do not fire inside identifiers (`lpx`, `xif(`), `let` needs its space
        ("lpx=1;xif(2);letter", ["=", ";", "(", ");"]),
        # type names attach only in type position (`:`/`@`), not as member/ident
        ("i=s:std.str;f=g(a:str;b:@str):str{<a}", ["i=", ":", ".str", ";", "f=", "(", ":str;", ":@str):str{<", "}"]),
    ],
)
def test_fragments(text: str, expected: list[str]) -> None:
    assert dmm.fragments(text) == expected


def test_fragments_roundtrip_covers_all_punctuation() -> None:
    text = 'f=pat(a:i64;b:@str):str{let r=mut.s.concat("_";a.get(0));<"\\(r)!"}'
    frags = dmm.fragments(text)
    # every non-atom character of the text is inside exactly one fragment (in order)
    joined = "".join(frags)
    stripped = "".join(c for c in text if not (c.isalnum() or c == "_"))
    for f in frags:  # fragments are substrings, in order
        assert f in text
    assert all(c in joined for c in stripped)


@pytest.mark.parametrize(
    "fragment, text, n",
    [
        ("lp(", "lp(x)help(lp(", 2),
        (".len", "a.len;b.length;c.len(", 2),  # `.len(` contains `.len` (only [a-z0-9] guards)
        ("i=", "i=io;xi=5;lp(let i=0;i<n;i=i+1)", 3),
        ("==", "a==b)==c", 2),
        ("\\(", '"_\\(a)\\(b)"', 2),
        ("mt ", "let v=mt f(x){$ok:v v}", 1),
    ],
)
def test_count_occurrences_is_boundary_guarded(fragment: str, text: str, n: int) -> None:
    assert dmm.count_occurrences(fragment, text) == n


# --- sample file ------------------------------------------------------------------


def test_sample_file_roundtrip(tmp_path: Path) -> None:
    meta = {"corpus": "~/x", "seed": 131, "requested": 2}
    rows = [("A-1", 'm=a;f=main():i64{io.println("_");<0};'), ("A-2", "m=b;")]
    p = tmp_path / "s.txt"
    dmm.write_sample(p, meta, rows)
    m2, r2 = dmm.read_sample(p)
    assert r2 == rows
    assert m2 == {"corpus": "~/x", "seed": "131", "requested": "2"}


def test_small_fixture_reads() -> None:
    meta, rows = dmm.read_sample(SMALL_SAMPLE)
    assert len(rows) == 40
    assert meta["seed"] == "131"
    assert all("\n" not in t for _, t in rows)


# --- synthetic catalogue -----------------------------------------------------------

_PAT = {
    "one/a.tk": "f=pat(xs:@i64):i64{let c=mut.0;lp(let i=0;i<xs.len;i=i+1){if(xs.get(i)%3==0){c=c+1}};<c}",
    "one/b.tk": "f=pat(xs:@i64):i64{<xs.len}",
    "two/c.tk": "f=pat(a:i64;b:i64):i64{<if(a>0||b>0){1}el{0}}",
    "three/d.tk": "f=pat(s:str):str{<s}",
}


def _catalogue(tmp_path: Path, entries: list[dict[str, Any]]) -> Path:
    p = tmp_path / "catalogue.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"protocol": "0.4", "entries": entries}), encoding="utf-8")
    return p


def _entry(eid: str, canonical: str, hot: str | None = None, forms: tuple[str, ...] = ("a", "b", "c")) -> dict[str, Any]:
    return {"id": eid,
            "candidates": [{"form": f, "fixture": f"patterns/{eid}/{f}.tk"} for f in forms],
            "verdict": {"canonical": canonical, "hot_path": hot, "status": "provisional"}}


def _stub_extract(toke_repo: Path) -> Any:
    def extract(fixture: Path) -> str:
        key = f"{fixture.parent.name}/{fixture.name}"
        if key not in _PAT:
            raise RuntimeError(f"function 'pat' not found in {key}")
        return _PAT[key]
    return extract


@pytest.fixture()
def synthetic(tmp_path: Path) -> dict[str, Any]:
    toke = tmp_path / "toke"
    for key in _PAT:
        f = toke / "patterns" / key
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("// fixture placeholder; extraction is stubbed\n")
    (toke / "patterns" / "two" / "b.tk").write_text("// present but has no pat\n")
    cat = _catalogue(tmp_path, [_entry("one", "a", "b"), _entry("two", "c"),
                                _entry("three", "d", None, ("d",))])
    meta, rows = dmm.read_sample(SMALL_SAMPLE)
    res = dmm.derive(dmm.load_catalogues([cat]), _stub_extract(toke), toke, dict(meta), rows,
                     dmm.sha256_file(SMALL_SAMPLE))
    return res


def _row(res: dict[str, Any], frag: str) -> dict[str, Any]:
    return next(r for r in res["fragments"] if r["fragment"] == frag)


def test_synthetic_forms_and_fragments(synthetic: dict[str, Any]) -> None:
    res = synthetic
    assert res["summary"] == {"forms_extracted": 4, "canonical_forms": 3, "hot_path_forms": 1,
                              "unparsed": 0, "forced": res["summary"]["forced"],
                              "verify": res["summary"]["verify"]}
    assert [(f["id"], f["role"], f["form"]) for f in res["forms"]] == [
        ("one", "canonical", "a"), ("one", "hot_path", "b"), ("two", "canonical", "c"),
        ("three", "canonical", "d")]
    assert res["forms"][0]["fragments"]["lp("] == 1
    assert res["forms"][0]["fragments"][".get("] == 1
    assert res["corpus_sample"]["records"] == 40
    assert res["corpus_sample"]["sample_file_sha256"] == dmm.sha256_file(SMALL_SAMPLE)


def test_synthetic_classes(synthetic: dict[str, Any]) -> None:
    res = synthetic
    frags = {r["fragment"] for r in res["fragments"]}
    # every D4 closed-class item is listed even when no canonical form contains it
    assert set(dmm.FORCED_D4) <= frags
    assert _row(res, "&&")["in_canonical_forms"] == []
    assert "not in any current canonical form" in _row(res, "&&")["rationale"]
    assert _row(res, "&&")["class"] == "forced"
    # f= is in all canonical forms; `.get(` only in one of three (verify); `.len` in hot path too
    assert _row(res, "f=") == {**_row(res, "f="), "class": "forced",
                               "in_canonical_forms": ["one", "three", "two"], "canonical_occurrences": 3}
    assert _row(res, ".get(")["class"] == "verify"
    assert _row(res, ".get(")["in_canonical_forms"] == ["one"]
    assert _row(res, ".len")["in_hot_path_forms"] == ["one"]
    assert _row(res, ".len")["in_canonical_forms"] == ["one"]
    # promotion: `){` appears in 2/3 canonical forms (>= 50%) and is common in the corpus;
    # `:str):str{<` is in 1/3 only -> verify even though the corpus has it
    assert _row(res, "){")["in_canonical_forms"] == ["one", "two"]
    assert _row(res, "){")["class"] == "forced"
    assert _row(res, ":str):str{<")["class"] == "verify"
    assert _row(res, "){")["rationale"].startswith("promoted")
    # single-character fragments are dropped; sigils excluded with zero occurrences
    assert not any(len(r["fragment"]) < 2 for r in res["fragments"])
    assert [e["fragment"] for e in res["excluded"]] == list(dmm.EXCLUDED_SIGILS)
    assert all(e["corpus_freq"] == 0 for e in res["excluded"])
    assert "$i64" not in frags
    # corpus numbers come from the hermetic sample (substring, guarded)
    texts = [t for _, t in dmm.read_sample(SMALL_SAMPLE)[1]]
    assert _row(res, "f=")["corpus_freq"] == sum(dmm.count_occurrences("f=", t) for t in texts)
    assert _row(res, "f=")["corpus_programs_share"] == 1.0
    # ordering: forced first, then by corpus frequency descending
    classes = [r["class"] for r in res["fragments"]]
    assert classes == sorted(classes, key=lambda c: c != "forced")
    freqs = [r["corpus_freq"] for r in res["fragments"] if r["class"] == "verify"]
    assert freqs == sorted(freqs, reverse=True)


def test_unparsed_fixtures_are_reported_not_silent(tmp_path: Path) -> None:
    toke = tmp_path / "toke"
    (toke / "patterns" / "two").mkdir(parents=True)
    (toke / "patterns" / "two" / "b.tk").write_text("// no pat\n")
    (toke / "patterns" / "two" / "a.blocked.tk").write_text("// blocked\n")
    cat = _catalogue(tmp_path, [_entry("two", "b", "a"), _entry("three", "a", None, ("a",))])
    meta, rows = dmm.read_sample(SMALL_SAMPLE)
    res = dmm.derive(dmm.load_catalogues([cat]), _stub_extract(toke), toke, dict(meta), rows, "0" * 64)
    reasons = {(u["id"], u["form"]): u["reason"] for u in res["unparsed"]}
    assert "RuntimeError: function 'pat' not found" in reasons[("two", "b")]
    assert "blocked: a.blocked.tk" in reasons[("two", "a")]
    assert "no file" in reasons[("three", "a")]
    assert res["summary"]["forms_extracted"] == 0
    assert res["summary"]["unparsed"] == 3
    assert {r["fragment"] for r in res["fragments"]} == set(dmm.FORCED_D4)


def test_multiple_catalogues_are_merged(tmp_path: Path) -> None:
    toke = tmp_path / "toke"
    for key in _PAT:
        f = toke / "patterns" / key
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("//\n")
    c1 = _catalogue(tmp_path / "w1", [_entry("one", "a")])
    c2 = _catalogue(tmp_path / "w2", [_entry("two", "c")])
    meta, rows = dmm.read_sample(SMALL_SAMPLE)
    res = dmm.derive(dmm.load_catalogues([c1, c2]), _stub_extract(toke), toke, dict(meta), rows, "0" * 64)
    assert [c["entries"] for c in res["catalogues"]] == [1, 1]
    assert _row(res, "f=")["in_canonical_forms"] == ["one", "two"]
    assert res["summary"]["canonical_forms"] == 2


def test_render_is_deterministic(synthetic: dict[str, Any]) -> None:
    assert dmm.render(synthetic) == dmm.render(json.loads(dmm.render(synthetic)))
    assert dmm.render(synthetic).endswith("\n")


# --- derivability (acceptance) ----------------------------------------------------


def _live() -> tuple[Path, Path] | None:
    try:
        tkc = tkcanon.find_tkc()
    except FileNotFoundError:
        return None
    toke = tkcanon.DEFAULT_TOKE_REPO
    if not (toke / "patterns" / "catalogue.json").is_file():
        return None
    return toke, tkc


def test_committed_list_is_derivable_from_committed_catalogues() -> None:
    """``data/must_merge_v04.json`` == fresh derivation (catalogue(s) + committed sample)."""
    live = _live()
    if live is None:
        pytest.skip("tkc / toke checkout not available")
    toke, tkc = live
    assert DATA.is_file() and SAMPLE.is_file()
    committed = DATA.read_text(encoding="utf-8")
    cats = dmm.load_catalogues(dmm.default_catalogues(toke))
    meta, rows = dmm.read_sample(SAMPLE)
    fresh = dmm.render(dmm.derive(cats, dmm.make_pat_extractor(toke, tkc), toke, dict(meta), rows,
                                  dmm.sha256_file(SAMPLE)))
    if fresh != committed:
        a, b = json.loads(committed), json.loads(fresh)
        diff = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
        pytest.fail(
            "data/must_merge_v04.json is not derivable from the current catalogue(s)/sample "
            f"(differing keys: {diff}); re-run scripts/derive_must_merge.py and commit -- never hand-edit"
        )


def test_committed_list_records_its_inputs() -> None:
    d = json.loads(DATA.read_text(encoding="utf-8"))
    assert d["story"] == "131.23"
    assert d["corpus_sample"]["seed"] == "131"
    assert d["corpus_sample"]["records"] >= 5000
    assert d["corpus_sample"]["sample_file_sha256"] == dmm.sha256_file(SAMPLE)
    assert d["unparsed"] == []
    assert {r["fragment"] for r in d["fragments"] if r["class"] == "forced"} >= set(dmm.FORCED_D4)
    assert all(e["corpus_freq"] == 0 for e in d["excluded"])
    live = _live()
    if live is not None:
        current = {c.name: dmm.sha256_file(c) for c in dmm.default_catalogues(live[0])}
        assert {c["path"]: c["sha256"] for c in d["catalogues"]} == current
