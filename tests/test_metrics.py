"""Story 131.20: metric orientation and gate tests (TEMSpec v1.0 §2).

These pin the three Phase-0 harness bugs from tokenizer-v04-plan.md §4:

1. ``char_to_token_ratio`` computed tokens/char but was gated ``<= 1.8`` as if
   it were chars/token -> the compression gate could never fail.
2. ``tokens_per_line`` on ``tkc --min`` (single-line) input is tokens/program.
3. "fertility" must be tokens per character (TEMSpec §2.4), lower is better.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from eval import (
    CHARS_PER_TOKEN_GATE_MIN,
    build_report,
    compression_gate,
    compute_chars_per_token,
    compute_corpus_fertility,
    compute_fertility,
    compute_tokens_per_line,
    compute_tokens_per_program,
    format_summary,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def _one_token_per_char(text: str) -> list[int]:
    """Worst-case tokenizer: every character is its own token."""
    return [ord(c) for c in text]


def _one_token_per_four_chars(text: str) -> list[int]:
    return [i for i in range(0, max(len(text), 1), 4)]


class TestFertilityOrientation:
    def test_fertility_is_tokens_per_char(self) -> None:
        # 8 chars, 2 tokens -> 0.25 tokens/char (NOT 4.0 chars/token)
        assert compute_fertility(["abcdefgh"], [[1, 2]]) == 0.25

    def test_lower_fertility_means_better_compression(self) -> None:
        text = "m=hello;f=main():i64{<0};"
        worse = compute_fertility([text], [_one_token_per_char(text)])
        better = compute_fertility([text], [_one_token_per_four_chars(text)])
        assert worse == 1.0
        assert better < worse

    def test_corpus_fertility_weights_by_length(self) -> None:
        programs = ["ab", "abcdefgh"]  # 2 + 8 = 10 chars
        tokens = [[1, 2], [3]]  # 2 + 1 = 3 tokens
        assert compute_corpus_fertility(programs, tokens) == 0.3
        # per-program mean is (1.0 + 0.125) / 2, different from the corpus ratio
        assert compute_fertility(programs, tokens) == pytest.approx(0.5625)

    def test_chars_per_token_is_inverse_of_corpus_fertility(self) -> None:
        programs = ["abcdefgh", "ijkl"]
        tokens = [[1, 2], [3]]
        cpt = compute_chars_per_token(programs, tokens)
        assert cpt == 4.0
        assert cpt == pytest.approx(1 / compute_corpus_fertility(programs, tokens))

    def test_empty_inputs(self) -> None:
        assert compute_fertility([], []) == 0.0
        assert compute_corpus_fertility([], []) == 0.0
        assert compute_chars_per_token([], []) == 0.0


class TestCompressionGateCanFail:
    """The pre-131.20 gate compared tokens/char (~0.3) against 1.8 -> always PASS."""

    def test_gate_fails_on_one_token_per_char(self) -> None:
        text = "m=hello;i=io:std.io;f=main():i64{io.println(\"_\");<0};"
        cpt = compute_chars_per_token([text], [_one_token_per_char(text)])
        assert cpt == 1.0
        assert compression_gate(cpt) is False

    def test_gate_passes_on_four_chars_per_token(self) -> None:
        text = "m=hello;i=io:std.io;f=main():i64{io.println(\"_\");<0};"
        toks = _one_token_per_four_chars(text)
        cpt = compute_chars_per_token([text], [toks])
        assert cpt >= CHARS_PER_TOKEN_GATE_MIN
        assert compression_gate(cpt) is True

    def test_old_orientation_would_have_passed_the_failing_case(self) -> None:
        # Regression guard: feeding tokens/char into the old ``<= 1.8`` check
        # passes even the worst tokenizer.  The new gate must reject it.
        text = "abcdefghij"
        fert = compute_corpus_fertility([text], [_one_token_per_char(text)])
        assert fert <= 1.8  # the bug: this always held
        assert compression_gate(compute_chars_per_token([text], [_one_token_per_char(text)])) is False

    def test_gate_rejects_zero_and_negative(self) -> None:
        assert compression_gate(0.0) is False
        assert compression_gate(-1.0) is False

    def test_gate_threshold_is_in_chars_per_token(self) -> None:
        assert CHARS_PER_TOKEN_GATE_MIN == 1.8
        assert compression_gate(1.8) is True
        assert compression_gate(1.79) is False
        assert compression_gate(2.5, minimum=3.0) is False


class TestTokensPerProgramVsPerLine:
    def test_multiline_source_differs(self) -> None:
        prog = "m=a;\nf=main():i64{\n<0\n};"  # 4 non-empty lines
        tpp = compute_tokens_per_program([_one_token_per_char(prog)])
        tpl = compute_tokens_per_line([prog], _one_token_per_char)
        assert tpp["mean"] == len(prog)
        assert tpl == (len(prog) - 3) / 4  # newlines are not in any line
        assert tpl != tpp["mean"]

    def test_min_canonical_form_degenerates_to_tokens_per_program(self) -> None:
        # tkc --min emits one program per line: tokens/line == tokens/program.
        prog = "m=a;f=main():i64{<0};"
        tpp = compute_tokens_per_program([_one_token_per_char(prog)])
        tpl = compute_tokens_per_line([prog], _one_token_per_char)
        assert tpl == tpp["mean"] == len(prog)

    def test_tokens_per_program_stats(self) -> None:
        stats = compute_tokens_per_program([[0] * 10, [0] * 20, [0] * 30])
        assert stats == {"mean": 20.0, "median": 20.0, "p95": 30.0}

    def test_tokens_per_line_ignores_blank_lines(self) -> None:
        assert compute_tokens_per_line(["ab\n\n\ncd"], _one_token_per_char) == 2.0
        assert compute_tokens_per_line([""], _one_token_per_char) == 0.0


class TestReportCarriesGate:
    def test_report_gate_fails_for_bad_tokenizer(self) -> None:
        programs = ["m=hello;f=main():i64{<0};"]
        toke = [_one_token_per_char(programs[0])]
        base = [_one_token_per_four_chars(programs[0])]
        report = build_report(programs, toke, base, vocab_size=256)
        assert report["fertility"] == 1.0
        assert report["chars_per_token"] == 1.0
        assert report["compression_gate_pass"] is False
        assert "FAIL" in format_summary(report)

    def test_report_gate_passes_for_good_tokenizer(self) -> None:
        programs = ["m=hello;f=main():i64{<0};"]
        toke = [_one_token_per_four_chars(programs[0])]
        base = [_one_token_per_char(programs[0])]
        report = build_report(programs, toke, base, vocab_size=256)
        assert report["compression_gate_pass"] is True
        assert "PASS" in format_summary(report)
        assert report["compression_ratio"] < 1.0


class TestRetrainBpeUsesFixedMetrics:
    def test_retrain_bpe_has_no_inverted_gate(self) -> None:
        spm = pytest.importorskip("sentencepiece")
        del spm
        import retrain_bpe

        assert not hasattr(retrain_bpe, "char_to_token_ratio")
        assert not hasattr(retrain_bpe, "tokens_per_line")

        class FakeSP:
            def encode(self, text: str, out_type: object = int) -> list[int]:
                return _one_token_per_char(text)

        metrics = retrain_bpe.compression_metrics(FakeSP(), ["abcdefgh", "ijkl"])
        assert metrics["fertility"] == 1.0
        assert metrics["chars_per_token"] == 1.0
        assert metrics["compression_gate_pass"] is False
        assert metrics["tokens_per_program_mean"] == 6.0
