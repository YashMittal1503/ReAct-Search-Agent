"""Small, dependency-free Okapi BM25 for ranking chunks within one research run.

The corpus is tiny (tens to a few hundred chunks), so a straightforward in-memory implementation is
fast and avoids an extra dependency.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")
STOPWORDS = frozenset(
    "a an and are as at be been but by can could did do does for from had has have how if in into is it its "
    "may might more most no not of on or our such than that the their them then there these they this those "
    "to was we were what when where which while who why will with would you your vs versus about".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens (keeps ``gpt-4o``, ``3.5`` intact), stopwords removed.

    >>> tokenize("The GPT-4o model costs $3.5 per 1M tokens, vs. Llama 3")
    ['gpt-4o', 'model', 'costs', '3.5', 'per', '1m', 'tokens', 'llama', '3']
    """
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    """Okapi BM25 (k1, b) over pre-tokenized documents."""

    def __init__(self, docs: list[list[str]], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(d) for d in docs]
        self.lengths = [len(d) for d in docs]
        self.avgdl = (sum(self.lengths) / len(docs)) if docs else 0.0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(docs)
        # BM25+-style smoothing keeps IDF positive even for very common terms.
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for doc, length in zip(self.docs, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / self.avgdl) if self.avgdl else self.k1
            s = 0.0
            for term in set(query):
                tf = doc.get(term, 0)
                if tf:
                    s += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / (tf + norm)
            out.append(s)
        return out
