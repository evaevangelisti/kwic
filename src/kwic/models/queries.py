"""
What a search is asked for.
"""

from dataclasses import dataclass

from .pos import POS


@dataclass(frozen=True, slots=True)
class Query:
    """
    One lemma to look for, narrowed to a part of speech or not.

    The engine cuts the lemma into words. One of several words is read as
    its words in order, and no tag narrows it: an established expression
    carries a category of its own, which none of its words need carry.

    Attributes:
        lemma: The dictionary form to look for, in whatever case.
        pos: The tag an occurrence of one word must carry, or None for
        any. A lemma of several words is not narrowed by it.
        forms: How else the lemma is written. A word written as one is taken
        even where the engine read it as another lemma.
    """

    lemma: str
    pos: POS | None = None
    forms: frozenset[str] = frozenset()
