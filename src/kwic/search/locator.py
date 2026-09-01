"""
The search a caller runs.
"""

from collections.abc import Iterable, Iterator, Sequence
from itertools import tee

from ..engines import Engine, SpacyEngine
from ..models import Context, Match, Query, Token
from .lexicon import Lexicon
from .reading import Reading

PARTICLE_WORDS = 2
"""How many words a phrasal verb is, its particle written apart or not."""


class Locator:
    """
    Locates the lemmas of a query in the context that attests them.

    A context is analysed rather than searched, so a search is as right as
    the reading behind it: a misread occurrence is a missed one.
    """

    def __init__(
        self,
        engine: Engine | None = None,
    ) -> None:
        """
        Take an engine to read with, loading the default one if given none.

        Args:
            engine: The analyser to read contexts with. The default loads
            spaCy's small English pipeline, which has to be installed.

        Raises:
            OSError: If the default pipeline is not installed.
        """
        self._engine: Engine = engine if engine is not None else SpacyEngine()

        self._last_queries: frozenset[Query] | None = None
        self._last_lexicon: Lexicon = Lexicon(lemmas={}, forms={}, counts=())

    def _lexicon(
        self,
        queries: frozenset[Query],
    ) -> Lexicon:
        """
        Gather the queries, holding on to the last gathering.

        Contexts are read one after another against the same queries, so a
        gathering is made once per set of them.

        Args:
            queries: The lemmas to look for, each narrowed to a tag or not.

        Returns:
            What a word is looked up in.
        """
        if queries != self._last_queries:
            self._last_queries = queries
            self._last_lexicon = Lexicon.gather(queries, self._engine.split_all)

        return self._last_lexicon

    @staticmethod
    def _match(
        span: Sequence[Token],
        index: int,
        text: str | None,
    ) -> Match:
        """
        Write down one occurrence, its range running from first word to last.

        Args:
            span: The words it is made of.
            index: Where the first of them falls among the words.
            text: The context, where the caller handed over one.

        Returns:
            The occurrence.
        """
        first_offsets, last_offsets = span[0].offsets, span[-1].offsets

        offsets = (
            None
            if first_offsets is None or last_offsets is None
            else (first_offsets[0], last_offsets[1])
        )

        return Match(
            lemma=" ".join(token.lemma for token in span),
            pos=span[0].pos,
            form=" ".join(token.form for token in span)
            if text is None or offsets is None
            else text[offsets[0] : offsets[1]],
            word_index=index,
            offsets=offsets,
        )

    @classmethod
    def _matches(
        cls,
        reading: Reading,
        lexicon: Lexicon,
        text: str | None,
    ) -> tuple[Match, ...]:
        """
        Read the queried lemmas off an analysed context.

        Where two lemmas open on one word the longer wins. A phrasal verb
        written apart holds the words between, so a range may hold another.

        Args:
            reading: The words of the context, laid out.
            lexicon: The lemmas to look for, gathered beforehand.
            text: The context, where the caller handed over one.

        Returns:
            The occurrences, leftmost first and one to a word.
        """
        matches: list[Match] = []
        cursor = 0

        while cursor < len(reading.places):
            span, step = cls._opening(reading, lexicon, cursor)

            if span is not None:
                matches.append(cls._match(span, reading.places[cursor], text))

            cursor += step

        return tuple(matches)

    @staticmethod
    def _opening(
        reading: Reading,
        lexicon: Lexicon,
        cursor: int,
    ) -> tuple[Sequence[Token] | None, int]:
        """
        Read the occurrence one word opens, the longest one winning.

        Args:
            reading: The words of the context, laid out.
            lexicon: The lemmas to look for.
            cursor: The word to read from.

        Returns:
            The words the occurrence is made of, or None where the word opens
            none, and how many words to read on from.
        """
        for count in lexicon.counts:
            if cursor + count <= len(reading.places):
                tags = reading.tags(lexicon, cursor, count)

                # A lemma of several words is an expression of its own, whose
                # category none of its words need carry.
                if tags is not None and (
                    count > 1 or reading.tokens[reading.places[cursor]].pos in tags
                ):
                    return reading.span(cursor, count), count

            # A phrasal verb is two words wherever its particle sits, so it is
            # read before the verb alone is read as a lemma.
            if count == PARTICLE_WORDS:
                parted = reading.phrasal_verb(lexicon, cursor)

                if parted is not None:
                    return parted, 1

        return None, 1

    def find_all(
        self,
        searches: Iterable[tuple[Context, Iterable[Query]]],
    ) -> Iterator[tuple[Match, ...]]:
        """
        Read many contexts, each for the lemmas asked of it.

        The contexts are handed to the engine as they come, so that a run
        over a stream of them holds one batch in memory rather than all.

        Args:
            searches: A context and the lemmas to look for in it, in pairs.

        Yields:
            The occurrences in one context, in the order the searches came
            in, and nothing at all for a context attesting none of them.
        """
        reading_searches, matching_searches = tee(searches)

        for tokens, (context, queries) in zip(
            self._engine.analyse_all(context for context, _ in reading_searches),
            matching_searches,
            strict=True,
        ):
            yield self._matches(
                Reading.read(tokens),
                self._lexicon(frozenset(queries)),
                context if isinstance(context, str) else None,
            )

    def find(
        self,
        context: Context,
        queries: Iterable[Query],
    ) -> tuple[Match, ...]:
        """
        Read every occurrence of the queried lemmas out of one context.

        The queries are gathered once and the words read once, so asking for
        one lemma and asking for ten thousand take the same pass.

        Args:
            context: The text to search, or the words it was split into.
            queries: The lemmas to look for, each narrowed to a tag or not.

        Returns:
            The occurrences, leftmost first.
        """
        return next(self.find_all(((context, queries),)))
