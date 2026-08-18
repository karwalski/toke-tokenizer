# toke-tokenizer

A pure Python BPE tokenizer for the [toke programming language](https://github.com/karwalski/toke). Trained on normalised toke source code with a 16,384-token vocabulary.

Achieves approximately 52% token reduction compared to OpenAI's cl100k_base tokenizer on toke source code.

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
