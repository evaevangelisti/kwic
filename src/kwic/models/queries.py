"""
What a search is asked for.
"""

from dataclasses import dataclass

from .pos import POS


@dataclass(frozen=True, slots=True)
class Query:
    """
    One lemma to look for, narrowed to a part of speech or not.

    The engine cuts the lemma into words, and a lemma of several of them is
    read as its words in order, under the tag the first one carries.

    Attributes:
        lemma: The dictionary form to look for, in whatever case.
        pos: The tag an occurrence must carry, or None for any.
    """

    lemma: str
    pos: POS | None = None
