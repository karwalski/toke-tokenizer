# Test fixtures

`spike_programs_min.txt` — 300 programs from the freeze-129 `regen_v04` corpus
(the 131.20 baseline sample, first 300 ids), canonicalised with `tkc --min`
(toke 2.8.0) and string-masked per plan D2. One program per line. Used by
`tests/test_hf_spike.py` / `scripts/hf_spike.py` to train a *toy* byte-level
tokenizer in a temp dir at test time.

No tokenizer artefact is stored here: the provenance rule (story 131.24,
`scripts/check_provenance.py`) covers `models/` and root `tokenizer*.json`;
toy models must be generated, never committed.
