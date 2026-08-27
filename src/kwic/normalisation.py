"""
Bringing a query and a lemma to the same spelling.
"""

import unicodedata

MARKS = str.maketrans({"’": "'", "‘": "'", "‐": "-", "‑": "-"})
"""Punctuation a text may set typographically. A context lemmatised to don’t
answers a query spelling don't, the two being one lemma."""


def normalise(
    lemma: str,
) -> str:
    """
    Bring one lemma to the spelling every other is compared in.

    Composing comes last: folding the case of a composed string may leave it
    decomposed, and two lemmas are equal here only if their code points are.

    Args:
        lemma: The lemma, as a query names it or an engine read it.

    Returns:
        It unified, folded and composed.
    """
    return unicodedata.normalize("NFC", lemma.translate(MARKS).casefold())
