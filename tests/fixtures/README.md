# Test fixtures

`spike_programs_min.txt` — 300 programs from the freeze-129 `regen_v04` corpus
(the 131.20 baseline sample, first 300 ids), canonicalised with `tkc --min`
(toke 2.8.0) and string-masked per plan D2. One program per line. Used by
`tests/test_hf_spike.py` / `scripts/hf_spike.py` to train a *toy* byte-level
tokenizer in a temp dir at test time.

No tokenizer artefact is stored here: the provenance rule (story 131.24,
`scripts/check_provenance.py`) covers `models/` and root `tokenizer*.json`;
toy models must be generated, never committed.

`must_merge_sample_small.txt` — the first 40 records of
`data/must_merge_sample_v04.txt` (story 131.23: 5,000 accepted `regen_v04`
records, `random.Random(131)`, `tkc --min` + D2 masking, `task_id<TAB>text`).
Hermetic corpus side of `tests/test_derive_must_merge.py`'s synthetic-catalogue
tests; the derivability test uses the full committed sample.
