"""
What a search hands back.
"""

from dataclasses import dataclass

from .annotations import Offsets
from .pos import POS


@dataclass(frozen=True, slots=True)
class Match:
    """
    One occurrence of a queried lemma.

    The lemma is the caller's own; everything else is the engine's reading.
    One of several words runs from the first of them to the last.

    Attributes:
        lemma: The lemma it was asked for, as the caller wrote it.
        pos: The tag it carries.
        form: How it is written.
        word_index: Where it opens among the words of the context, from zero.
        offsets: Where it falls in the text, or None where the caller handed
        over words rather than a text.
    """

    lemma: str
    pos: POS
    form: str
    word_index: int
    offsets: Offsets | None
