"""
What a search looks a word up in.
"""

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Self

from ..models import POS, Query
from ..normalisation import normalise, spellings

ALL_TAGS = frozenset(POS)
"""What a query naming no part of speech narrows to, which is no narrowing."""

type Cut = Callable[[Iterable[str]], Iterator[tuple[str, ...]]]
"""How an engine cuts a spelling into the words it would read."""


@dataclass(frozen=True, slots=True)
class Lexicon:
    """
    The queried lemmas and forms, cut into words and normalised.

    Attributes:
        lemmas: The tags each queried lemma is read under.
        forms: The tags each queried form is read under.
        counts: How many words a queried spelling holds, longest first.
    """

    lemmas: Mapping[tuple[str, ...], frozenset[POS]]
    forms: Mapping[tuple[str, ...], frozenset[POS]]
    counts: Sequence[int]

    @classmethod
    def gather(
        cls,
        queries: Iterable[Query],
        cut: Cut,
    ) -> Self:
        """
        Gather the queries into what one pass over a context reads.

        Args:
            queries: The lemmas to look for, each narrowed to a tag or not.
            cut: How the engine cuts a spelling into words.

        Returns:
            The lemmas and the forms to look for, every tag standing in where
            a query named none.
        """
        asked = list(queries)

        lemmas = _keys(
            [
                (query, spelling)
                for query in asked
                for spelling in spellings(query.lemma)
            ],
            cut,
        )

        forms = _keys(
            [
                (query, spelling)
                for query in asked
                for form in query.forms
                for spelling in spellings(form)
            ],
            cut,
        )

        return cls(
            lemmas=lemmas,
            forms=forms,
            counts=sorted({len(key) for key in (*lemmas, *forms) if key}, reverse=True),
        )


def _keys(
    spelled: Sequence[tuple[Query, str]],
    cut: Cut,
) -> dict[tuple[str, ...], frozenset[POS]]:
    """
    Cut every spelling into the words a context would be read as.

    Args:
        spelled: A query and one way of writing what it looks for.
        cut: How the engine cuts a spelling into words.

    Returns:
        The tags to look each cut up under, the whole spelling standing
        beside the cut for a text that writes it as one word.
    """
    keys: dict[tuple[str, ...], frozenset[POS]] = {}

    for (query, spelling), words in zip(
        spelled, cut(spelling for _, spelling in spelled), strict=True
    ):
        tags = ALL_TAGS if query.pos is None else frozenset({query.pos})
        written = tuple(normalise(word) for word in words if not word.isspace())

        for key in (written, (normalise(spelling),)):
            keys[key] = keys.get(key, frozenset()) | tags

    return keys
