---
language: [en]
license: apache-2.0
tags: [toke, tokenizer, bpe, code, v0.3, historical]
---

<!--
  Model card for https://huggingface.co/karwalski/toke-tokenizer, prepared by story
  132.4. The live card still says "52% average token reduction vs cl100k_base" and
  "19 tokens (vs 49 cl100k)" — both withdrawn (toke/docs/metrics-baseline.md §
  Canonical wording, § What is withdrawn outright). Uploading this file as the
  repository README is an owner action; see
  toke/docs/about/registry-descriptions.md.
-->

# toke tokenizer — v0.3 BPE (16,384 vocab)

The **v0.3** BPE tokenizer for the [toke programming language](https://tokelang.dev):
16,384 tokens, trained on v0.3-syntax toke source. toke is on **v0.4**; the v0.4 retrain
is tracked as Epic 116.9 and has not shipped. Use this artefact to reproduce v0.3
measurements, not as "the toke tokenizer".

## About toke

> toke: a compiled language designed for LLM code generation, with a small grammar, one
> canonical form and compiler verification.

toke is a compiled programming language designed for LLM code generation. It has 14
keywords, a 59-character set, a backtrack-free grammar with bounded lookahead, and one
canonical form per construct, chosen by measurement in a 46-pattern catalogue and
reproduced by `tkc --min`. That makes generated code cheap to constrain during decoding,
cheap for a compiler to verify afterwards, and compact to emit. Token efficiency is one
measured property of toke, always reported with its tokenizer and its baseline, not the
whole claim.

- Website: [tokelang.dev](https://tokelang.dev)
- Compiler, specification and standard library:
  [github.com/karwalski/toke](https://github.com/karwalski/toke)
- This tokenizer:
  [github.com/karwalski/toke-tokenizer](https://github.com/karwalski/toke-tokenizer)

*The one-liner and the paragraph above are reproduced word for word from the canonical
description,
[`docs/about/canonical.md`](https://github.com/karwalski/toke/blob/main/docs/about/canonical.md).
Every number published about toke comes from
[`docs/metrics-baseline.md`](https://github.com/karwalski/toke/blob/main/docs/metrics-baseline.md)
and nowhere else.*

## The "52% reduction" claim is withdrawn

This repository previously advertised "52% average token reduction vs cl100k_base across
42 benchmark programs". That figure compared **two tokenizers on one text** (Toke-16K
v0.3 vs cl100k_base, both encoding the same toke source, N = 42) — it was never a
comparison with Python — and it is superseded on two counts:

- On canonical v0.4 `tkc --min` text (N = 2,000 stratified corpus records, 2026-09-18)
  every shipped toke tokenizer is *worse* than cl100k_base; the 8K SentencePiece needs
  **15.4% more** tokens.
- This v0.3 vocabulary's apparent margin (ratio 0.545) is inflated by an `unk_token` of
  `null`, which silently deletes every backslash — 2,606 of them in that sample.

**Token efficiency, measured:** under one shared tokenizer (cl100k_base) toke costs
**1.34× [1.22, 1.48]** the tokens of equivalent Python on the 60 Gate-1 tasks (N = 60,
2026-09-19) — more, not fewer. The v0.3-era "52% fewer tokens" figure was a
*tokenizer-vs-tokenizer* measurement on identical toke text (Toke-16K v0.3 vs cl100k_base,
N = 42) and is superseded: on canonical v0.4 text the shipped 8K tokenizer needs **15.4%
more** tokens than cl100k_base (N = 2,000). See `docs/metrics-baseline.md`.

## Facts

| Property | Value |
|---|---|
| Vocabulary | 16,384 |
| Syntax generation | toke **v0.3** (superseded by v0.4) |
| Training data | v0.3-syntax toke source, string contents replaced with a placeholder |
| Known defect | `unk_token` is `null`; backslashes are dropped on encode |
| Licence | Apache-2.0 |

## This is not the model's tokenizer

The published model [`karwalski/toke`](https://huggingface.co/karwalski/toke) uses Qwen's
151K-vocab tokenizer internally. This artefact measures how toke source *could* be
tokenised by a toke-native model; no such model has been trained.

## Usage

```python
from tokenizers import Tokenizer

tok = Tokenizer.from_file("tokenizer_v03.json")
print(len(tok.encode('m=fib;f=fib(n:i64):i64{if(n<2){<n};<fib(n-1)+fib(n-2)};').ids))
```

Never tokenise Python or any non-toke source with this tokenizer for a comparison: it is
trained on toke text and off-domain it measures its own training bias. A cross-language
number uses **one** tokenizer on both sides.
