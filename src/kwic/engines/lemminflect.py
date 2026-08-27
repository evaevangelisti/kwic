"""
Analysis by spaCy for the tags and by LemmInflect for the lemmas.
"""

from typing import cast, override

from lemminflect import (  # pyright: ignore[reportMissingTypeStubs]
    getLemma,  # pyright: ignore[reportUnknownVariableType]
)
from spacy.tokens import Token as Word

from ..models import POS
from .spacy import SpacyEngine

SUPPORTED = frozenset({POS.ADJ, POS.ADV, POS.NOUN, POS.PROPN, POS.VERB})
"""The tags LemmInflect holds lemmas under. A word carrying any other is left
to the rules of the pipeline, which read is as be and n't as not."""


class LemmInflectEngine(SpacyEngine):
    """
    A search taking its tags from a spaCy pipeline and its lemmas elsewhere.

    A lexicon holds every reading of a form and no way of choosing between
    them: leaves under NOUN is both leave and leaf, and the likelier wins.
    """

    @override
    def _lemma(
        self,
        word: Word,
    ) -> str:
        """
        Look one word up under the tag the pipeline gave it.

        Args:
            word: The word the pipeline read.

        Returns:
            The likeliest lemma the lexicon holds for it, or what the rules
            of the pipeline made of it where the tag is not one it holds.
        """
        if word.pos_ not in SUPPORTED:
            return super()._lemma(word)

        lemmas = cast("tuple[str, ...]", getLemma(word.text, word.pos_))

        return lemmas[0] if lemmas else word.text
