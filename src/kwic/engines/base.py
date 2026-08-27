"""
What every engine hands a search, whatever it wraps underneath.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator

from ..models import Context, Token


class Engine(ABC):
    """
    An analyser reading contexts into the words a search runs over.

    What it promises is the tagset and the offsets, not the reading: two
    engines put to one context may lemmatise it differently.
    """

    @abstractmethod
    def analyse_all(
        self,
        contexts: Iterable[Context],
    ) -> Iterator[tuple[Token, ...]]:
        """
        Read every context into the words it is made of.

        Args:
            contexts: The texts to read, or the words they were split into.

        Yields:
            The words of one context, in the order the contexts came in.
        """

    def analyse(
        self,
        context: Context,
    ) -> tuple[Token, ...]:
        """
        Read one context into the words it is made of.

        Args:
            context: The text to read, or the words it was split into.

        Returns:
            Its words, in the order they are written.
        """
        return next(self.analyse_all((context,)))

    def split_all(
        self,
        texts: Iterable[str],
    ) -> Iterator[tuple[str, ...]]:
        """
        Cut every text into words the way this engine cuts a context.

        A query names a lemma, and its words have to be the words the engine
        would read: pre-university is one word to a space and three to spaCy.

        Args:
            texts: The texts to cut.

        Yields:
            The words of one text, in the order they are written.
        """
        for tokens in self.analyse_all(texts):
            yield tuple(token.form for token in tokens)
