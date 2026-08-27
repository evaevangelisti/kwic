"""
What an engine reads a context into.
"""

from dataclasses import dataclass

from .pos import POS

type Offsets = tuple[int, int]
"""Half-open range of code points, as Python slices them."""


@dataclass(frozen=True, slots=True)
class Token:
    """
    One word of a context, as an engine read it.

    Attributes:
        lemma: Its dictionary form, spelled as the engine spells it.
        pos: The tag it carries here.
        form: The word as it is written.
        offsets: Where it falls in the text, or None where the caller handed
        over words rather than a text.
        particle_of: Where the verb it is the particle of falls among the
        words, or None where it is not one. Only a parser sets it.
    """

    lemma: str
    pos: POS
    form: str
    offsets: Offsets | None
    particle_of: int | None = None
