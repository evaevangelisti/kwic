"""
What an engine made of one context, laid out to be read.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Self

from ..models import Token
from ..normalisation import normalise
from .lexicon import Asked, Lexicon

PARAGRAPH = 2
"""How many line breaks part two blocks rather than wrap one line."""


@dataclass(frozen=True, slots=True)
class Reading:
    """
    What an engine made of one context, laid out for one pass over it.

    Whitespace is a word to spaCy and to nobody else, so the words are held
    by the places they fall in, and a blank line stands among them.

    Attributes:
        tokens: The words as the engine read them.
        places: Where each word to be read falls among them.
        lemmas: What each of those was read as, normalised.
        forms: How each of those is written, normalised.
        particles: Where the particles hanging off a word fall, under it.
    """

    tokens: Sequence[Token]
    places: Sequence[int]
    lemmas: Sequence[str]
    forms: Sequence[str]
    particles: Mapping[int, Sequence[int]]

    @classmethod
    def read(
        cls,
        tokens: Sequence[Token],
    ) -> Self:
        """
        Lay out what an engine handed back.

        Args:
            tokens: The words of the context, in order.

        Returns:
            The same words, ready to be looked up.
        """
        places = [
            index
            for index, token in enumerate(tokens)
            if not token.form.isspace() or token.form.count("\n") >= PARAGRAPH
        ]

        particles: dict[int, list[int]] = {}

        for index, token in enumerate(tokens):
            if token.particle_of is not None:
                particles.setdefault(token.particle_of, []).append(index)

        return cls(
            tokens=tokens,
            places=places,
            lemmas=[normalise(tokens[index].lemma) for index in places],
            forms=[normalise(tokens[index].form) for index in places],
            particles=particles,
        )

    def asked(
        self,
        lexicon: Lexicon,
        cursor: int,
        count: int,
    ) -> Asked | None:
        """
        Look a span of words up, by what it was read as or by how it is written.

        Args:
            lexicon: The lemmas and the forms to look for.
            cursor: Where the span opens.
            count: How many words it holds.

        Returns:
            What the span was asked for, or None where nothing asks for it.
        """
        asked = lexicon.lemmas.get(tuple(self.lemmas[cursor : cursor + count]))

        if asked is None:
            return lexicon.forms.get(tuple(self.forms[cursor : cursor + count]))

        return asked

    def span(
        self,
        cursor: int,
        count: int,
    ) -> Sequence[Token]:
        """
        Take the words one span is made of.

        Args:
            cursor: Where it opens.
            count: How many words it holds.

        Returns:
            The words themselves.
        """
        return [self.tokens[index] for index in self.places[cursor : cursor + count]]

    def phrasal_verb(
        self,
        lexicon: Lexicon,
        cursor: int,
    ) -> tuple[Sequence[Token], str] | None:
        """
        Look for a phrasal verb whose particle is written apart from it.

        Args:
            lexicon: The lemmas to look for.
            cursor: Where the verb falls among the words to be read.

        Returns:
            The verb and its particle with the lemma they were asked for, or
            None where the two are not a lemma anyone asked for.
        """
        verb = self.places[cursor]

        for particle in self.particles.get(verb, ()):
            asked = lexicon.lemmas.get(
                (self.lemmas[cursor], normalise(self.tokens[particle].lemma))
            )

            if asked is not None:
                return (self.tokens[verb], self.tokens[particle]), asked.lemma

        return None
