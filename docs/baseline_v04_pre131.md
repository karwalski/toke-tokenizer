# Tokenizer baseline v0.4 — `pre131`

Story 131.20 (plan D7). Generated 2026-09-18T13:04:10.445652+00:00. **This is the PRE-rewrite baseline**; 131.21 re-runs the same script with `--label post131` over the same ids.

## Headline (masked canonical text)

| tokenizer | vocab | total tokens | tokens/program mean [95% CI] | median | p95 | fertility tok/char (mean) [95% CI] | chars/token | vocab util. | vs cl100k ratio [95% CI] | roundtrip | lossy chars |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tokenizer_v03 | 16384 | 131,998 | 66.0 [64.6, 67.4] | 60 | 124 | 0.2238 [0.2220, 0.2256] | 4.447 | 27.3% | 0.5445 [0.5401, 0.5488] | 55.0% | 2,606 |
| sp8k | 8000 | 279,672 | 139.8 [137.2, 142.5] | 130 | 258 | 0.4775 [0.4755, 0.4795] | 2.099 | 29.3% | 1.1536 [1.1470, 1.1601] | 100.0% | 0 |
| sp32k | 32768 | 279,143 | 139.6 [136.9, 142.2] | 128 | 258 | 0.4781 [0.4761, 0.4799] | 2.103 | 9.0% | 1.1515 [1.1457, 1.1571] | 6.5% | 13,605 |
| qwen2.5-coder | 151665 | 250,287 | 125.1 [122.7, 127.6] | 113 | 228 | 0.4296 [0.4273, 0.4319] | 2.345 | 1.7% | 1.0324 [1.0307, 1.0341] | 100.0% | 0 |
| cl100k_base | 100277 | 242,427 | 121.2 [118.9, 123.6] | 110 | 221 | 0.4155 [0.4135, 0.4174] | 2.422 | 2.8% | 1.0000 [1.0000, 1.0000] | 100.0% | 0 |
| o200k_base | 200019 | 245,217 | 122.6 [120.2, 125.0] | 112 | 224 | 0.4197 [0.4177, 0.4217] | 2.394 | 1.4% | 1.0115 [1.0107, 1.0123] | 100.0% | 0 |

Ratio < 1 means fewer tokens than cl100k_base on the same text (TEMSpec §2.2 compression-ratio form; same source language, different tokenizers — informational cross-tokenizer density, §6.2, not the §6.1 gate metric). `lossy chars` = characters dropped (HF file with `unk_token: null`) or mapped to `<unk>` (SentencePiece without byte fallback); a tokenizer with lossy chars > 0 under-counts and its row is informational only.

Lossy tokenizers — most frequent dropped/unk surfaces (surface, count):

- **tokenizer_v03**: `'\\'`×2584, `'^'`×18, `'~'`×4
- **sp32k**: `'_'`×5822, `'@'`×3021, `'\\'`×2191, `'$'`×1537, `'_\\'`×393, `'&&'`×373, `'%'`×213, `'&'`×32, `'^'`×18, `'&~'`×2, `'~'`×2, `'@@'`×1

## Unmasked (informational, plan D2)

| tokenizer | total tokens | tokens/program mean | fertility (mean) | chars/token |
|---|---:|---:|---:|---:|
| tokenizer_v03 | 158,760 | 79.4 | 0.2558 | 3.850 |
| sp8k | 288,333 | 144.2 | 0.4726 | 2.120 |
| sp32k | 289,746 | 144.9 | 0.4765 | 2.109 |
| qwen2.5-coder | 258,223 | 129.1 | 0.4270 | 2.367 |
| cl100k_base | 249,721 | 124.9 | 0.4122 | 2.447 |
| o200k_base | 252,786 | 126.4 | 0.4167 | 2.418 |

## Per-category (masked): tokens/program mean

| category | n | tokenizer_v03 | sp8k | sp32k | qwen2.5-coder | cl100k_base | o200k_base |
|---|---:|---:|---:|---:|---:|---:|---:|
| A-ARR | 147 | 52.8 | 126.9 | 131.3 | 122.1 | 118.3 | 120.1 |
| A-CND | 147 | 65.4 | 123.7 | 120.2 | 126.6 | 118.8 | 120.2 |
| A-ERR | 132 | 92.3 | 175.0 | 175.5 | 162.3 | 157.1 | 160.1 |
| A-MTH | 147 | 52.1 | 111.2 | 108.1 | 112.6 | 106.9 | 108.9 |
| A-SRT | 147 | 86.5 | 191.7 | 198.3 | 188.8 | 183.4 | 186.3 |
| A-STR | 147 | 73.4 | 154.0 | 150.2 | 130.8 | 128.3 | 129.4 |
| D-CFG | 136 | 60.0 | 131.2 | 131.1 | 105.4 | 104.2 | 104.9 |
| D-CLI | 139 | 62.8 | 126.5 | 125.7 | 105.6 | 103.4 | 103.9 |
| D-CRY | 146 | 61.3 | 121.3 | 118.0 | 116.4 | 107.9 | 109.6 |
| D-DAT | 145 | 71.0 | 157.1 | 159.8 | 134.3 | 130.9 | 132.6 |
| D-FIO | 132 | 59.8 | 132.8 | 131.9 | 108.4 | 106.6 | 107.5 |
| D-NET | 147 | 66.9 | 147.5 | 146.0 | 120.0 | 117.2 | 117.9 |
| D-TST | 146 | 59.8 | 129.5 | 128.2 | 110.0 | 107.4 | 108.1 |
| D-WEB | 142 | 61.3 | 130.7 | 130.9 | 107.6 | 105.9 | 106.4 |

## Per-category (masked): fertility (corpus tokens/char)

| category | tokenizer_v03 | sp8k | sp32k | qwen2.5-coder | cl100k_base | o200k_base |
|---|---:|---:|---:|---:|---:|---:|
| A-ARR | 0.1909 | 0.4589 | 0.4750 | 0.4418 | 0.4279 | 0.4343 |
| A-CND | 0.2468 | 0.4667 | 0.4534 | 0.4777 | 0.4484 | 0.4534 |
| A-ERR | 0.2390 | 0.4532 | 0.4546 | 0.4206 | 0.4070 | 0.4147 |
| A-MTH | 0.2227 | 0.4754 | 0.4621 | 0.4815 | 0.4571 | 0.4654 |
| A-SRT | 0.2101 | 0.4658 | 0.4819 | 0.4587 | 0.4456 | 0.4526 |
| A-STR | 0.2273 | 0.4771 | 0.4654 | 0.4052 | 0.3976 | 0.4008 |
| D-CFG | 0.2233 | 0.4883 | 0.4880 | 0.3924 | 0.3879 | 0.3907 |
| D-CLI | 0.2439 | 0.4911 | 0.4881 | 0.4099 | 0.4014 | 0.4034 |
| D-CRY | 0.2425 | 0.4797 | 0.4667 | 0.4601 | 0.4267 | 0.4335 |
| D-DAT | 0.2191 | 0.4850 | 0.4932 | 0.4145 | 0.4041 | 0.4095 |
| D-FIO | 0.2153 | 0.4782 | 0.4751 | 0.3906 | 0.3839 | 0.3873 |
| D-NET | 0.2219 | 0.4892 | 0.4844 | 0.3979 | 0.3889 | 0.3912 |
| D-TST | 0.2245 | 0.4863 | 0.4813 | 0.4132 | 0.4032 | 0.4059 |
| D-WEB | 0.2286 | 0.4872 | 0.4881 | 0.4013 | 0.3947 | 0.3965 |

## Per-difficulty (masked): tokens/program mean

| difficulty | n | tokenizer_v03 | sp8k | sp32k | qwen2.5-coder | cl100k_base | o200k_base |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 267 | 54.2 | 115.2 | 114.7 | 104.5 | 100.7 | 102.0 |
| 2 | 1316 | 63.4 | 135.8 | 135.4 | 120.7 | 117.1 | 118.4 |
| 3 | 417 | 81.7 | 168.3 | 168.6 | 152.3 | 147.3 | 149.0 |

## v0.4 syntax single-token coverage (plan D4 forced list)

From `docs/eval_syntax_tokens_v04_pre131.json` (2000 canonical programs). `standalone` = pattern alone encodes to one piece; `unsplit` = in-context occurrences that lie inside one token (fragment merges count); `exact` = occurrences that are a token by themselves.

| model | vocab | standalone single-token | in-context occurrences | unsplit | exact | split occurrences |
|---|---:|---:|---:|---:|---:|---:|
| `models/toke.model` | 8000 | 15/26 (57.7%) | 33,675 | 58.42% | 45.82% | 14,003 |
| `models/32k/toke.model` | 32768 | 3/26 (11.5%) | 33,675 | 16.21% | 3.38% | 28,215 |
| `tokenizer_v03.json` | 16384 | 16/26 (61.5%) | 33,675 | 85.14% | 14.22% | 5,005 |

Patterns with zero occurrences in the sample (listed in the story but absent from v0.4 corpus text): `$i64`, `$f64`, `$str`, `$bool`, `$u64`, `$byte`.

## D4 AddedToken substring wart (`m= f= t= i=`)

From `docs/wart_d4_pre131.json` over 23,381 canonical programs (6,834,145 chars; 1 failed `--min`).

- boundary-aligned `[mfti]=` occurrences (decl heads + `;i=i+1` loop steps): **127,096**
- wart occurrences (identifier char + `[mfti]` + `=`, e.g. `result=`, `sum=`, `xi=`): **21,722** = 17.091% of boundary-aligned; followed by `=` 20,034 / `==` 1,688
- programs touched: 9,142 (39.1%); 3.1785 per 1k chars

| head | boundary-aligned | wart | wart / aligned |
|---|---:|---:|---:|
| `m=` | 23,598 | 1,608 | 6.81% |
| `f=` | 52,752 | 159 | 0.3% |
| `t=` | 1,636 | 19,094 | 1167.11% |
| `i=` | 49,110 | 861 | 1.75% |

Realised on `models/toke.model` (user_defined_symbols): 21,722 of 166,220 emitted head pieces (13.07%) are preceded by an identifier character.

**Recommendation: `seeded_merges`** (material=True; rule: wart/aligned ≥ 5% or programs touched ≥ 10%).

## Tokenizers

| name | kind | version | identity |
|---|---|---|---|
| tokenizer_v03 | hf-json | tokenizers 0.22.2 | `sha256:42631f3afddbc9c911770cffbf14e25a59b43485c6e4b12ff93a56dab1330a74` |
| sp8k | sentencepiece | sentencepiece 0.2.1 | `sha256:9a754777212d340b29cd566512bea6d4c68a20ee6e21353905a548769f15de5a` |
| sp32k | sentencepiece | sentencepiece 0.2.1 | `sha256:a98ccbdb6181fb3099a1db29f6b59a341e5e3dcfdcbbfa35a71f7405c69fedf0` |
| qwen2.5-coder | hf-hub | transformers 5.8.1 | `Qwen/Qwen2.5-Coder-7B` |
| cl100k_base | tiktoken | tiktoken 0.12.0 | `cl100k_base` |
| o200k_base | tiktoken | tiktoken 0.12.0 | `o200k_base` |
| llama3 | hf-hub | — | SKIPPED: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files. |

## Methodology

- Sample: 2000 records, ids file `data/baseline_sample_ids_v04.txt` (sha256 `d83702ea2ed6c5fa4fc258e75ac6afc5507f27a202842484c02b95c19583ea4c`), stratified by category × difficulty, seed 131, drawn from the freeze-129 `MANIFEST.jsonl`.
- Corpus source: `/tmp/t131/freeze_sample` (records extracted from ~/tk/archive/toke-corpus-regen_v04-freeze129-20260819/regen_v04-freeze129.tar.zst (sha256 a00dc0a1…cde9, verified with shasum -c), the 131.12 freeze-129 snapshot; not the live toke-corpus tree).
- Record integrity: 2000/2000 record files match the freeze-129 sha256 in the ids file.
- Records dropped (failed `tkc --min`): 0.
- Canonical form: `tkc --min` (toke 2.8.0), one program per line; strings masked to `_` per plan D2 (`toke/scripts/patterns/mask_strings.py`, keeps `\(...)` interpolation interiors).
- sample sha256 (sha256 over the sorted per-record `min_sha256`s): `0739cf6c8303c1bebd5417104649f34fcf27fe008516d958e8f5e2a3d14526d6`; masked-text sha256: `042390903c5c3b0f321f3f349bd6d1f518b308be5596af3db444c3151fb51014`.
- Token counts: `len(encode(text))` with default settings (TEMSpec §3.3); Qwen adds no BOS/EOS by default.
- CIs: percentile bootstrap, 10,000 resamples, seed 131 (TEMSpec §5.2). Fertility = tokens/char (§2.4). Vocab utilisation = unique ids used / vocab size (§2.5).
- Per-record counts: `data/baseline_v04_pre131_per_record.csv` (TEMSpec §5.3).


## Findings and decisions (131.20, hand-written; appended via `--notes`)

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
