"""The deterministic embedder, shared by the tape generator and the embed server.

One source of truth so the manifest's ground-truth retrievals and the embeddings
the pipeline actually indexes are computed by the SAME code. See the spec in the
rgen.py module docstring; the short version is a hashed bag-of-words cosine:
lowercase ASCII tokens of length >= 2, minus a fixed stopword list, hashed with
FNV-1a into a 512-dim signed accumulator, L2-normalised. It captures lexical
overlap, which is all the deterministic oracle needs; the live mode swaps in a
real semantic embedding model behind the same HTTP contract.
"""

import math
import re

D = 512

STOPWORDS = {
    "the", "a", "an", "to", "of", "and", "or", "is", "are", "be", "in", "on",
    "for", "with", "as", "at", "by", "it", "its", "this", "that", "these",
    "those", "we", "you", "do", "does", "how", "what", "when", "which", "if",
    "then", "from", "into", "out", "up", "our", "your", "not", "no", "can",
    "will", "has", "have", "was", "were", "run", "use", "using",
}

_TOKEN = re.compile(r"[a-z0-9]+")


def fnv1a64(s: str) -> int:
    h = 0xCBF29CE484222325
    for b in s.encode("ascii", "ignore"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def tokenize(text: str):
    return [t for t in _TOKEN.findall(text.lower())
            if len(t) >= 2 and t not in STOPWORDS]


def embed(text: str):
    v = [0.0] * D
    for tok in tokenize(text):
        h = fnv1a64(tok)
        sign = 1.0 if (h >> 63) & 1 else -1.0
        v[h % D] += sign
    norm = math.sqrt(sum(x * x for x in v))
    if norm > 0.0:
        v = [x / norm for x in v]
    return v


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))
