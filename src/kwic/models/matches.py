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

    The lemma and the tag are the engine's reading of it, not the query's
    wording. One of several words runs from the first of them to the last.

    Attributes:
        lemma: The lemma it was read as, its words spaced apart.
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
