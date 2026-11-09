"""Quote verification and Scroll-to-Text-Fragment deep links.

A *text fragment* (``https://site/page#:~:text=start,end``) makes Chromium-based browsers (and
Safari 16.1+) scroll to and highlight the matching passage. We build one from the first and last
few words of the supporting quote, after verifying the quote really appears in the source text,
so a click lands on the exact passage that backs the citation.

Spec: https://wicg.github.io/scroll-to-text-fragment/
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from urllib.parse import quote as _pct

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n{1,}")
_MD_NOISE_RE = re.compile(r"(\*\*|__|`|^#+\s*|^\s*[-*+]\s+|^\s*>\s*|\[([^\]]*)\]\([^)]*\))", re.M)
_WS_RE = re.compile(r"\s+")
_QUOTE_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                            "–": "-", "—": "-", " ": " "})


def clean_markdown(text: str) -> str:
    """Strip Markdown syntax so the quote matches the *rendered* page text."""
    text = _MD_NOISE_RE.sub(lambda m: m.group(2) if m.group(2) is not None else "", text)
    text = text.replace("|", " ")
    return _WS_RE.sub(" ", text).strip()


def _norm(text: str) -> str:
    """Comparison form: NFKC, ASCII-ish punctuation, lowercase, collapsed whitespace."""
    text = unicodedata.normalize("NFKC", text).translate(_QUOTE_MAP)
    return _WS_RE.sub(" ", clean_markdown(text)).strip().lower()


def verify_quote(quote: str, source_text: str, *, min_ratio: float = 0.72) -> tuple[str, bool]:
    """Ground ``quote`` in ``source_text``.

    Returns ``(passage, verified)``:

    * exact (normalized) substring match -> the cleaned quote, ``True``;
    * otherwise the most similar sentence of the source if its similarity is at least
      ``min_ratio`` -> that sentence, ``True`` (the LLM paraphrased slightly);
    * otherwise the best-effort sentence (or the original quote) with ``False``.
    """
    cleaned_quote = clean_markdown(quote).strip(" \"'")
    if not cleaned_quote or not source_text:
        return cleaned_quote, False

    nq, ns = _norm(cleaned_quote), _norm(source_text)
    if nq and nq in ns:
        return cleaned_quote, True

    best_sentence, best_ratio = "", 0.0
    for sentence in _SENTENCE_SPLIT_RE.split(source_text):
        sentence = clean_markdown(sentence)
        if len(sentence) < 20:
            continue
        matcher = difflib.SequenceMatcher(None, nq, _norm(sentence), autojunk=False)
        if matcher.real_quick_ratio() < best_ratio or matcher.quick_ratio() < best_ratio:
            continue
        ratio = matcher.ratio()
        if ratio > best_ratio:
            best_sentence, best_ratio = sentence, ratio

    if best_ratio >= min_ratio:
        return best_sentence, True
    if best_ratio >= 0.5:
        return best_sentence, False
    return cleaned_quote, False


def _encode_fragment_term(term: str) -> str:
    """Percent-encode a text-fragment term; ``-``, ``,`` and ``&`` are spec delimiters."""
    return _pct(term, safe="").replace("-", "%2D")


def build_text_fragment(url: str, quote: str, *, edge_words: int = 4) -> str:
    """Append a ``#:~:text=`` directive highlighting ``quote`` to ``url``.

    Short quotes (``<= 2 * edge_words`` words) are encoded whole; longer ones use the
    ``textStart,textEnd`` range form with the first/last ``edge_words`` words, which is robust to
    minor differences in the middle of the passage.

    >>> build_text_fragment("https://ex.com/a", "Monoliths are simpler to deploy and debug early on")
    'https://ex.com/a#:~:text=Monoliths%20are%20simpler%20to,and%20debug%20early%20on'
    """
    # Escape parentheses so the link survives inside Markdown ``[text](url)`` syntax.
    base = url.split("#", 1)[0].replace("(", "%28").replace(")", "%29")
    words = clean_markdown(quote).strip(" \"'").split()
    # Trailing punctuation often differs between the quote and the page; drop it from the edges.
    words = [w for w in words if w.strip(".,;:!?\"'()[]")]
    if not words:
        return base
    if len(words) <= 2 * edge_words:
        directive = _encode_fragment_term(" ".join(words).rstrip(".,;:!?"))
    else:
        start = " ".join(words[:edge_words])
        end = " ".join(words[-edge_words:]).rstrip(".,;:!?")
        directive = f"{_encode_fragment_term(start)},{_encode_fragment_term(end)}"
    return f"{base}#:~:text={directive}"
