"""Small, dependency-free Okapi BM25 used for the lexical half of hybrid search.

Dense embeddings are good at paraphrase; BM25 is good at exact identifiers,
codes, names and numbers ("clause 14.2", "SKU-4471") that embeddings blur.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_TOKEN = re.compile(r"\w+", re.UNICODE)
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have how i in is it of on or that the this "
    "to was were what when where which who why will with do does did can you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.term_freqs = [Counter(tokenize(doc)) for doc in documents]
        self.lengths = np.array([sum(tf.values()) for tf in self.term_freqs], dtype=np.float32)
        self.avg_length = float(self.lengths.mean()) if len(documents) else 0.0
        doc_freq: Counter[str] = Counter()
        for tf in self.term_freqs:
            doc_freq.update(tf.keys())
        n = len(documents)
        self.idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in doc_freq.items()}

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.term_freqs), dtype=np.float32)
        if not self.term_freqs or self.avg_length == 0:
            return scores
        terms = [t for t in set(tokenize(query)) if t in self.idf]
        if not terms:
            return scores
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avg_length)
        for term in terms:
            tf = np.array([freqs.get(term, 0) for freqs in self.term_freqs], dtype=np.float32)
            scores += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return scores
