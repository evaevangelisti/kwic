"""
Bringing a query and a lemma to the same spelling.
"""

import re
import unicodedata
from itertools import product

MARKS = str.maketrans({"’": "'", "‘": "'", "‐": "-", "‑": "-", "–": "-"})
"""Punctuation a text may set typographically. A context lemmatised to don’t
answers a query spelling don't, the two being one lemma."""

SEPARATORS = re.compile(r"[\s-]+")
"""What parts the words of a lemma, whichever space or hyphen is written."""

PARTED = 4
"""How many words a lemma may hold before the plain spellings alone are
written, the ways of parting them doubling with every word."""


def spellings(
    lemma: str,
) -> set[str]:
    """
    Write one lemma the ways its words may be parted.

    A dictionary writes above-board and a text aboveboard, and the tokeniser
    cuts the two differently, so every parting is looked for.

    Args:
        lemma: The lemma, as a query names it.

    Returns:
        Every way of parting its words with a space or a hyphen, and the one
        that parts them with nothing.
    """
    written = " ".join(lemma.translate(MARKS).split())
    parts = [part for part in SEPARATORS.split(written) if part]

    if len(parts) > PARTED:
        return {written, " ".join(parts), "-".join(parts), "".join(parts)}

    return {
        "".join(
            part + parting for part, parting in zip(parts, (*partings, ""), strict=True)
        )
        for partings in product(" -", repeat=len(parts) - 1)
    } | {"".join(parts)}


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
