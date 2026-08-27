"""
The engine the suite reads with, which reads nothing at all.
"""

from collections.abc import Iterable, Iterator
from typing import override

from kwic import Context, Engine, Token


class Given(Engine):
    """
    An engine handing back a reading settled beforehand.

    A property about a search has to hold whatever the engine made of a
    context. Settling the reading is what lets one be stated, and no model is
    loaded to read it.
    """

    def __init__(
        self,
        reading: tuple[Token, ...],
    ) -> None:
        """
        Take the reading to hand back.

        Args:
            reading: The words of every context to come.
        """
        self._reading: tuple[Token, ...] = reading

    @override
    def analyse_all(
        self,
        contexts: Iterable[Context],
    ) -> Iterator[tuple[Token, ...]]:
        """
        Hand back what was written down, once per context.

        Args:
            contexts: The contexts, which are counted and not read.

        Yields:
            The reading settled for each of them.
        """
        for _ in contexts:
            yield self._reading

    @override
    def split_all(
        self,
        texts: Iterable[str],
    ) -> Iterator[tuple[str, ...]]:
        """
        Cut every text on its spaces, no reading being settled for a query.

        Args:
            texts: The texts to cut.

        Yields:
            The words of one text, in the order they are written.
        """
        for text in texts:
            yield tuple(text.split())
