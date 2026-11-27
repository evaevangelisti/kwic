"""
What a search is asked for.
"""

from dataclasses import dataclass

from .pos import POS


@dataclass(frozen=True, slots=True)
class Query:
    """
    One lemma to look for, optionally with a part of speech.

    The engine cuts the lemma into words. The tag of a multi-token match
    describes the expression as a whole.

    Attributes:
        lemma: The dictionary form to look for, in whatever case.
        pos: The required token tag or the declared expression tag, if given.
        forms: Alternate spellings to match even if the engine reads another lemma.
    """

    lemma: str
    pos: POS | None = None
    forms: frozenset[str] = frozenset()
