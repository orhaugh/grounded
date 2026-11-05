"""The deterministic embedder, shared by the tape generator and the embed server.

One source of truth so the manifest's ground-truth retrievals and the embeddings
the pipeline actually indexes are computed by the SAME code. See the spec in the
rgen.py module docstring; the short version is a hashed bag-of-words cosine:
lowercase ASCII tokens of length >= 2, minus a fixed stopword list, hashed with
FNV-1a into a 512-dim signed accumulator, L2-normalised. It captures lexical
overlap, which is all the deterministic oracle needs. A real semantic embedding
model could serve the same HTTP contract instead; nothing here checks that.
"""

import math
import re

D = 512

# How far clink's score for a chunk may sit from the reference's. clink scores
# cosine similarity in float32 (SimSIMD); the reference computes it in float64
# from the same unit vectors. A float32 dot product of n terms is within
# n * u of the exact value for unit vectors, u = 2**-24, so for n = 512 within
# 3.05e-5; the kernel's own normalisation by float32 norms can add as much
# again, and rounding the inputs to float32 adds 2u, 1.2e-7: 6.1e-5 in all.
# SCORE_TOL rounds that bound up. Two chunks whose reference scores are more
# than 2 * SCORE_TOL apart cannot change places in clink's ranking; two that
# are closer can, which is what "tied" means in the manifest and the oracle.
SCORE_TOL = 1e-4

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
