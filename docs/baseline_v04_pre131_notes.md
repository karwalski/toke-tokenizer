## Findings and decisions (131.20, hand-written; appended via `--notes`)

**Basis for every number below** (TEMSpec §6.3): N = 2,000 stratified records from the
2026-08-19 corpus freeze, canonical `tkc --min` text with string bodies masked to `"_"`;
ids in `data/baseline_sample_ids_v04.txt`. All comparisons are *one text, several
tokenizers* (§2.2 compression ratio) — none of them is a comparison with another
language.

### Headline reading
- **Every shipped toke tokenizer is worse than cl100k_base on canonical v0.4 text.** `sp8k` (`models/toke.model`, the TEMSpec `toke-bpe-8k`) needs **15.4% more** tokens than cl100k_base (ratio 1.154, CI [1.147, 1.160]); `sp32k` the same (1.152). Qwen2.5-Coder needs 3.2% more than cl100k; o200k 1.2% more. The v0.3-era "12.5% better than cl100k" prior does not survive `--min` + masking (plan D7 predicted this).
- `tokenizer_v03.json` *appears* to win (ratio 0.545) but is **lossy on v0.4 code**: its model has `unk_token: null` and no `\` or `^` in the vocab, so HF `tokenizers` silently drops every `\` — i.e. every `\(...)` interpolation opener and every escape — and it has no decoder (generic decode joins pieces with spaces). Its numbers are informational only; it cannot be the pre/post reference and must not ship as the runtime vocab. `sp32k` is also lossy (13,605 `<unk>` tokens: no byte fallback, trained on the 80-char syntax).
- The Phase-3 gate therefore anchors on **cl100k_base = 242,427 tokens / 121.2 mean / fertility 0.4155** on this sample; the v0.4 tokenizer must beat that by the margin Phase 3 sets, measured with `scripts/baseline_v04.py --label post131` over the *same* ids.

### v0.4 syntax gap (item 3)
- On the D4 forced list, `sp8k` leaves 41.6% of in-context occurrences split across tokens (14,003 of 33,675); `sp32k` splits 83.8%; `tokenizer_v03` splits 14.9% but is lossy on `\(`.
- The `$i64 $f64 $str $bool $u64 $byte` type sigils named in the plan/story have **zero occurrences** in v0.4 canonical text: v0.4 writes types as `:i64` / `@i64` and uses `$` only for variant tags (`$ok:` / `$err:`). The D4 forced list should replace the sigils with `:i64 :str :bool :f64 :u64 @i64 @str` (verification group in `eval_syntax_tokens.py`) and add `$ok:` / `$err:`.
- Expression-`if` is `if(` … `el{` in canonical text; match is `mt ` (space-separated scrutinee); early return is `{<` / `;<`.

### D4 AddedToken wart (item 5) — decision: **seeded merges, not AddedTokens**
- Over the full freeze (23,381 canonical programs): 21,722 wart occurrences vs 127,096 boundary-aligned heads = **17.1%**, touching **39.1% of programs**. `t=` dominates (19,094: `result=`, `count=`, `out=`, `cnt=`, `left=`, `right=`…); `m=` 1,608 (`sum=`, `num=`); `i=` 861 (`xi=`, `mi=`); `f=` 159 (`buf=`).
- On the shipped `sp8k` (which already uses `user_defined_symbols` = AddedToken semantics) 13.1% of emitted `[mfti]=` pieces are mid-identifier — the wart is real today, not hypothetical.
- Both material thresholds (≥ 5% ratio, ≥ 10% of programs) are exceeded by a wide margin: Phase 2 must seed `m= f= t= i=` as ordinary merges (so BPE only forms them at natural boundaries) rather than as AddedTokens. `==`, `!=`, `&&`, `||`, `<=`, `>=`, `\(`, `@(`, `lp(` have no such identifier-suffix hazard and can stay AddedTokens if needed, but with byte-level BPE they are expected to merge naturally.

### Qwen alignment (item 2) — deliverable for Epic 128
- Proper run (transformers 5.8.1, `Qwen/Qwen2.5-Coder-7B` from the HF cache): set overlap 31.5% / novel 68.5% after `▁`/`Ġ` normalisation (the 9.8.2 "100% novel" figure was the `qwen_vocab_size: 0` artefact).
- **Occurrence-weighted coverage 83.3%**: on canonical code, 5 of 6 toke-8k token occurrences are already single Qwen tokens. Uncovered mass is concentrated in `i= f= m= t=` (Qwen has no `x=` merges), `.get(`, and toke fragment merges (`64{`, `+1){`, `<0};`). Verdict `use_qwen_tokenizer_directly` (coverage ≥ 80%); Qwen also needs 10.5% *fewer* tokens than `sp8k` on this text. Report: `docs/alignment/`.

### HF byte-level spike (item 4)
- Toy D1-config tokenizer (ByteLevel, `add_prefix_space=False`, `use_regex=False`, ByteLevel decoder, 512 vocab, 300 programs) trained and saved in a temp dir; `tests/test_hf_spike.py` covers the three consumers.
- `toke-model/train/train_1b.py::load_tokenizer` is still `raise NotImplementedError`; it needs exactly `tokenizers.Tokenizer.from_file(path / "tokenizer.json")` (verified to load and round-trip). Its module cannot even be imported outside the toke-model package layout (`train.config`), so the test skips when that is the case.
- The pip runtime (`python/toke_tokenizer/tokenizer.py`) loads the byte-level JSON but its char-level `encode` cannot round-trip (strict xfail). Adding the GPT-2 byte↔unicode map on the way in and out (`hf_spike.ByteLevelShim`, ~15 lines) makes it round-trip **and reproduce HF's ids exactly** on 100/100 programs — that is the Phase-4 change for `toke_tokenizer` 0.2.0.
- Viz: raw pieces (`Ġio`) render correctly through `piece_to_text`; `decode(encode(x)) == x` incl. non-ASCII; a multi-byte char split across byte tokens decodes per-id to U+FFFD, so the website viz must decode token *runs* (or use offsets), never single ids.

### Harness / corpus findings to file as stories
- **`tkc --min` is not single-line for string literals containing raw newlines**: `D-DAT-0003v219` (compiles, exit 0) emits 8 lines because its `let nl="⏎"` literal is preserved verbatim. Plan D6 assumes one program per line "by construction"; either `--min` should emit `\n` escapes inside string bodies or the corpus validator must reject raw newlines in literals. 1 of 23,382 freeze records; 0 in the 2,000 sample (dropped + logged by the harness).
- **`corpus/regen_v04/MANIFEST.jsonl` `sha256` (sha of `tk_source`) is stale for 227 of the 2,000 sampled records** (11.4%) — not refreshed after the 129.4-5 repair pass. The freeze-129 record-file SHAs (`regen/freeze/freeze_129_manifest.jsonl`) are correct (2000/2000 match) and are what the ids file carries.
- `tkc --min a.tk b.tk` (multiple files) hangs ~80 s and emits one line; the harness calls it per file (≈11 ms each, thread pool).
- `tokenizer_v03.json` lossiness (above) also means the website token-viz baked from it undercounts on any program with interpolation.

### Skipped / blocked
- **llama3**: `meta-llama/Meta-Llama-3-8B` is gated and not in the HF cache (no HF token configured); skipped, row recorded as such. Add it for Phase 3 once a token is available (the runner picks it up automatically).
- Network was available; Qwen2.5-Coder-7B was served from the HF cache (`local_files_only`).
