# v0.4 must-merge list (story 131.23)

`data/must_merge_v04.json` is the syntax must-merge list for the v0.4 tokenizer,
**derived** from the pattern catalogue (`toke/patterns/catalogue*.json`, story
131.7) by `scripts/derive_must_merge.py`. It is never hand-edited:
`tests/test_derive_must_merge.py::test_committed_list_is_derivable_from_committed_catalogues`
re-derives it from the committed catalogue(s) and the committed corpus sample and
asserts byte identity.

## What is in it

For every catalogue entry the **canonical** form and (when set) the **hot_path**
form are `tkc --min`'d, the `f=pat` function is isolated (131.4 `count_tokens`)
and string-masked (plan D2), and the text is split into *syntax fragments* —
maximal punctuation runs with type names absorbed in type position, plus
standalone keyword / member / declaration / result-arm heads. The exact grammar
is the `scripts/derive_must_merge.py` module docstring (`grammar_version: 1`).

Each fragment row carries:

| field | meaning |
|---|---|
| `in_canonical_forms` / `in_hot_path_forms` | catalogue ids whose form contains the fragment |
| `canonical_occurrences` | occurrences across the canonical `pat` functions |
| `corpus_freq` / `corpus_programs` / `corpus_programs_share` | substring occurrences (identifier-boundary guarded) over the 5,000-record seeded corpus sample |
| `class` | `forced` or `verify` (see below) |
| `rationale` | why it has that class |

Classes (plan §3 D4 **as amended by 131.20**):

* **forced** — the closed-class list: operators `== != && || <= >=`, `@(`, `\(`,
  `lp(`, `mt `, `if(`, `el{`, and the declaration heads `m= f= t= i=` as
  **seeded merges, not AddedTokens** (131.20: the AddedToken substring wart hits
  17.1% of heads / 39.1% of programs, `t=` dominating). Plus any fragment in
  ≥ 50% of canonical forms with corpus programs share ≥ 5% (currently `let `,
  `){`). Closed-class items are listed even when no canonical form contains
  them (`&&`, `<=`, `!=`, `m=`, `i=`, `t=` — the `pat` function is not a whole
  program).
* **verify** — every other extracted fragment (`.get(`, `.len`, `mut.`, `};`,
  `):i64{`-style merges, `$ok:`/`$err:` …): must merge *naturally*; Phase 3
  checks it.
* **excluded** (separate key) — the D4 type sigils `$i64 $f64 $str $bool $u64
  $byte`: **zero occurrences** in v0.4 text (131.20; confirmed 0/5,000 here).
  Types are `:i64` / `@i64`; `$` only opens `$ok:` / `$err:` and user type names.

Single-character fragments are dropped (every byte is a base token).

## Current numbers (catalogue.json 19 + catalogue.wave2.json 27 entries; 46 canonical + 6 hot-path forms; 0 unparsed)

18 forced / 117 verify. Corpus sample: 5,000 accepted `regen_v04` records,
`random.Random(131)`, `tkc --min` (toke 2.8.0) + mask, 0 `--min` failures;
stored in `data/must_merge_sample_v04.txt` (sha256 recorded in the data file).

Top 20 by corpus frequency (share = fraction of the 5,000 programs containing it):

| fragment | class | canonical forms | corpus freq | share |
|---|---|---:|---:|---:|
| `};` | verify | 5/46 | 21,040 | 100.0% |
| `);` | verify | 10/46 | 15,448 | 81.7% |
| `let ` | forced (promoted) | 25/46 | 14,764 | 80.9% |
| `){` | forced (promoted) | 31/46 | 13,934 | 88.6% |
| `i=` | forced (D4 head) | 0/46 | 13,799 | 97.3% |
| `")` | verify | 1/46 | 11,719 | 65.0% |
| `f=` | forced (D4 head) | 46/46 | 11,348 | 100.0% |
| `))` | verify | 2/46 | 11,250 | 67.9% |
| `if(` | forced | 17/46 | 9,325 | 82.2% |
| `;<` | verify | 1/46 | 7,741 | 94.3% |
| `\(` | forced | 5/46 | 6,648 | 46.1% |
| `");` | verify | 3/46 | 5,848 | 45.8% |
| `"}` | verify | 1/46 | 5,750 | 45.8% |
| `.len` | verify | 17/46 | 5,481 | 62.1% |
| `"\(` | verify | 2/46 | 5,346 | 42.8% |
| `m=` | forced (D4 head) | 0/46 | 5,291 | 100.0% |
| `.get(` | verify | 22/46 | 5,002 | 46.7% |
| `==` | forced | 7/46 | 4,559 | 51.2% |
| `));` | verify | 1/46 | 4,276 | 30.5% |
| `};<` | verify | 5/46 | 4,185 | 72.2% |

Rare D4 items for the record: `&&` 906 (11.2%), `>=` 902, `<=` 781, `mt ` 777
(8.1%), `||` 765, `t=` 442, `!=` 434 (7.2%). `$ok:` / `$err:` 759 / 758 (8.0%).

Verify fragments with **zero** corpus occurrences (wave-2 library idioms the
regen corpus does not use yet: `.sort(`, `.filter(`, `.reduce(`, `.keys(`,
`.read(`, `.write(`, `.dec(`, `.i64(`, `.str(`, `@("":"");` …) stay listed —
the Phase 3 gate should treat `corpus_freq == 0` as "cannot be verified on the
holdout", not as a failure.

## How it feeds 116.9

* **Phase 2 (seeding)** — `train_v04.py` reads `class == "forced"` and seeds
  those strings as merges (D4). Do **not** register `m= f= t= i=` as
  `AddedToken`s. The sigils under `excluded` must not be seeded.
* **Phase 3 (gates)** — every `forced` fragment must be a single token in
  context (`scripts/eval_syntax_tokens.py` `exact`); every `verify` fragment
  with `corpus_programs_share ≥ 5%` must be *unsplit* in context (inside one
  token, larger merges allowed); rarer `verify` rows are reported, not gated.
  Use `corpus_freq` to set the per-fragment support threshold for the D4 n-gram
  verification pass.

## Re-running

```
python3 scripts/derive_must_merge.py \
    --corpus ~/tk/toke-corpus/corpus/regen_v04 --sample 5000 --seed 131
```

draws a fresh sample and rewrites both `data/must_merge_sample_v04.txt` and
`data/must_merge_v04.json`; without `--corpus` it re-derives from the committed
sample (what the test does). Re-run **after every catalogue change** (wave 2
merges, verdict flips from 131.25 / 131.26) and after the corpus rewrite lands
(131.19 → the sample must be redrawn from the rewritten corpus, same seed).
Commit script, sample and data file together; the test fails otherwise.
