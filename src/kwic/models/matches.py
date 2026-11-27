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

    The lemma is the caller's own. An expression runs from its first token
    to its last.

    Attributes:
        lemma: The lemma it was asked for, as the caller wrote it.
        pos: The token or expression tag, or None if unknown or ambiguous.
        form: How it is written.
        word_index: Where it opens among the words of the context, from zero.
        offsets: Text range, or None for pretokenised contexts.
    """

    lemma: str
    pos: POS | None
    form: str
    word_index: int
    offsets: Offsets | None
