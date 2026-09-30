"""The v0.3 vocabulary ships in the tree (story 132.109).

Until 132.109 ``python/toke_tokenizer/data/tokenizer_v03.json`` was untracked, so a
fresh clone had the ``toke_tokenizer`` code but not the file it loads and
``count_tokens()`` failed.  It is tracked now (1.4 MiB, under the 5 MB limit of
AGENTS §4.4), and the root ``tokenizer_v03.json`` -- the path that
``scripts/baseline_v04.py``, the provenance record and callers in other repos name --
is a symlink to it, so there is one copy of the bytes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PACKAGED = REPO / "python" / "toke_tokenizer" / "data" / "tokenizer_v03.json"
ROOT_PATH = REPO / "tokenizer_v03.json"
RECORD = REPO / "tokenizer_v03.provenance.json"


def test_packaged_vocab_matches_its_provenance_record() -> None:
    record = json.loads(RECORD.read_text())
    assert hashlib.sha256(PACKAGED.read_bytes()).hexdigest() == record["sha256"]


def test_root_path_is_the_packaged_file() -> None:
    assert ROOT_PATH.is_symlink()
    assert ROOT_PATH.resolve() == PACKAGED.resolve()


def test_count_tokens_loads_the_packaged_vocab() -> None:
    sys.path.insert(0, str(REPO / "python"))
    try:
        from toke_tokenizer import count_tokens, decode, encode
    finally:
        sys.path.pop(0)
    src = "m=main;f=main():i64{<0};"
    assert count_tokens(src) > 0
    assert len(encode(src)) == count_tokens(src)
    assert isinstance(decode(encode(src)), str)
