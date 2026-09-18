# Tokenizer Alignment Report (toke vs Qwen2.5-Coder)

Generated: 2026-09-18T12:53:32.511088+00:00  
toke model: `models/toke.model` (kind=sentencepiece, sha256 `9a754777212d340b…`)  
Qwen model: `Qwen/Qwen2.5-Coder-7B` (transformers 5.8.1)  
Sample: 500 canonical (`tkc --min` + masked) programs, seed 42

## Vocabulary overlap (pieces normalised to surface text)

| Metric | Value |
|---|---|
| toke vocab (normalised, non-special) | 7910 |
| Qwen vocab (normalised, non-special) | 151651 |
| Intersection | 2492 |
| Jaccard | 0.0159 |
| Overlap (intersection / toke vocab) | 31.5% |
| Novel toke tokens | 5418 (68.5%) |

## Occurrence-weighted coverage on canonical code

- toke token occurrences: 70533
- covered by a single Qwen token: 58738 (**83.3%**)

Top uncovered toke pieces (surface text, count):

```
    1443  'i='
    1126  'f='
     560  'm='
     501  't='
     483  '.get('
     462  '64{'
     412  '+1){'
     313  '0;'
     306  '64):'
     257  '<0};'
     241  '64;'
     167  '.0;'
     157  '{<'
     150  '";"'
     135  ')};'
      93  '=0;'
      91  '1;'
      85  '"};'
      79  '64{<0};'
      77  '64):@'
      73  '{<""};'
      67  '=0){'
      62  '}};'
      56  'lookuperr'
      54  '=0){<'
      52  '):@'
      52  ');<'
      51  '};<'
      51  '){<'
      50  '"}};'
      48  ')};<'
      44  'coun'
      41  ' resul'
      40  '5;'
      40  '."";'
      39  '1:'
      39  '")){'
      38  '3;'
      37  'delim'
      37  '<0){'
```

## Per-sample tokenization

- mean toke tokens/program: 141.1
- mean Qwen tokens/program: 127.5
- mean Qwen/toke ratio: 0.92x

## Recommendation

**Action:** `use_qwen_tokenizer_directly` (decision variable: occurrence_coverage_pct = 83.3%)

83.3% of toke token occurrences on canonical code are already single Qwen tokens (68.5% of toke vocab entries are novel by set overlap). Qwen's tokenizer can be used directly; extension is optional.

Thresholds: coverage >= 80% -> use Qwen directly; 50-80% -> targeted extension; < 50% -> extension prototype.  Set-overlap novelty is reported for continuity with the 9.8.2 run but is not the decision variable (it counts every rare merge equally).
