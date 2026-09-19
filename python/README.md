# toke-tokenizer

A pure Python BPE tokenizer for the [toke programming language](https://tokelang.dev).

**This package is the v0.3 tokenizer, not a current one.** The vocabulary shipped here is
the 16,384-token BPE trained on **v0.3-syntax** toke source. toke is on **v0.4**; the v0.4
retrain is tracked as Epic 116.9 and has not shipped. Use this package to reproduce v0.3
measurements and to count tokens against that vocabulary — not as "the toke tokenizer".

## About toke

> toke: a compiled language designed for LLM code generation, with a small grammar, one
> canonical form and compiler verification.

- Website: [tokelang.dev](https://tokelang.dev)
- Compiler, specification and standard library:
  [github.com/karwalski/toke](https://github.com/karwalski/toke)
- This tokenizer:
  [github.com/karwalski/toke-tokenizer](https://github.com/karwalski/toke-tokenizer)

*The one-liner above is reproduced word for word from the canonical description,
[`docs/about/canonical.md`](https://github.com/karwalski/toke/blob/main/docs/about/canonical.md).
Every number published about toke comes from
[`docs/metrics-baseline.md`](https://github.com/karwalski/toke/blob/main/docs/metrics-baseline.md)
and nowhere else.*

**Token-efficiency claims — read before quoting.** This package previously advertised
"approximately 52% token reduction compared to cl100k_base". That claim is **withdrawn**.
It described Toke-16K encoding toke source versus cl100k_base encoding *the same toke
source* (one text, two tokenizers, N = 42 v0.3 benchmark programs) — never a comparison
with Python — and it no longer holds: on canonical `tkc --min` v0.4 text (N = 2,000
stratified corpus records) every shipped toke tokenizer needs **more** tokens than
cl100k_base, and this v0.3 vocabulary's apparent margin is inflated by an `unk_token` of
`null` that silently drops backslashes. Use it to count tokens, not to make a claim.

Two rules for anyone measuring with it:

1. **Never tokenize Python (or any non-toke source) with this tokenizer** for a
   comparison. It is trained on toke text; off-domain it inflates the other side and
   measures its own training bias.
2. **A cross-language number uses one tokenizer on both sides.** Under cl100k_base on
   both sides, toke currently costs about 1.3x the tokens of equivalent Python
   (1.34x [1.22, 1.48], N = 60 Gate-1 tasks).

Current measurements and the approved wording for any public claim:
[`docs/metrics-baseline.md`](https://github.com/karwalski/toke/blob/main/docs/metrics-baseline.md).

## Installation

```bash
pip install toke-tokenizer
```

## Usage

```python
from toke_tokenizer import encode, decode, count_tokens

# Tokenize toke source code
text = "let x:int = 42"
tokens = encode(text)
print(tokens)        # list of token IDs

# Decode back to text
original = decode(tokens)
print(original)      # "let x:int = 42"

# Count tokens
n = count_tokens(text)
print(f"{n} tokens")
```

### String normalisation

Toke source often contains string literals that are not useful for structural tokenization. You can normalise strings before counting:

```python
from toke_tokenizer import count_tokens

# Without normalisation
count_tokens('let msg:str = "hello world"')

# With normalisation (replaces string contents with "_")
count_tokens('let msg:str = "hello world"', normalise_strings=True)
# Equivalent to counting: let msg:str = "_"
```

## API

- `encode(text: str) -> list[int]` — Tokenize text into a list of token IDs.
- `decode(ids: list[int]) -> str` — Convert token IDs back to text.
- `count_tokens(text: str, normalise_strings: bool = False) -> int` — Count the number of tokens in text.

## Details

- Vocabulary: 16,384 tokens
- Algorithm: Byte-Pair Encoding (BPE)
- Pre-tokenization: splits on newlines (each newline is a separate token)
- Special tokens: `<|endoftext|>` (0), `<pad>` (1), `<newline>` (2)

## License

MIT
