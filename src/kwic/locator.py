"""
The search a caller runs.
"""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from itertools import tee

from .engines import Engine, SpacyEngine
from .models import POS, Context, Match, Query, Token
from .normalisation import normalise

type Lexicon = Mapping[tuple[str, ...], frozenset[POS]]
"""What a search looks a word up in: every queried lemma, cut into its words
and normalised, under the tags it is read beneath."""

ALL_TAGS = frozenset(POS)
"""What a query naming no part of speech narrows to, which is no narrowing."""


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
        self._last_lexicon: Lexicon = {}

    def _gather_queries(
        self,
        queries: frozenset[Query],
    ) -> Lexicon:
        """
        Gather the queries into what one pass over a context reads.

        Contexts are read one after another against the same queries, so the
        last gathering is held on to and made once per set of them.

        Args:
            queries: The lemmas to look for, each narrowed to a tag or not.

        Returns:
            The tags to read each lemma under, every tag standing in where a
            query named none.
        """
        if queries == self._last_queries:
            return self._last_lexicon

        ordered_queries = list(queries)
        split_lemmas = self._engine.split_all(query.lemma for query in ordered_queries)

        lexicon: dict[tuple[str, ...], frozenset[POS]] = {}

        for query, words in zip(ordered_queries, split_lemmas, strict=True):
            lemma = tuple(normalise(word) for word in words if not word.isspace())
            tags = ALL_TAGS if query.pos is None else frozenset({query.pos})

            lexicon[lemma] = lexicon.get(lemma, frozenset()) | tags

        self._last_queries = queries
        self._last_lexicon = lexicon

        return lexicon

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

    @staticmethod
    def _phrasal_verb(
        tokens: Sequence[Token],
        index: int,
        lemma: str,
        particles: Sequence[int],
        lexicon: Lexicon,
    ) -> Sequence[Token] | None:
        """
        Look for a phrasal verb whose particle is written apart from it.

        Args:
            tokens: The words of the context.
            index: Where the verb falls among them.
            lemma: What the verb was read as, normalised.
            particles: Where the particles hanging off it fall.
            lexicon: The lemmas to look for.

        Returns:
            The verb and its particle, or None where the two are not a lemma
            anyone asked for.
        """
        for particle in particles:
            tags = lexicon.get((lemma, normalise(tokens[particle].lemma)))

            if tags is not None and tokens[index].pos in tags:
                return (tokens[index], tokens[particle])

        return None

    @classmethod
    def _matches(
        cls,
        tokens: Sequence[Token],
        lexicon: Lexicon,
        text: str | None,
    ) -> tuple[Match, ...]:
        """
        Read the queried lemmas off an analysed context.

        Where two lemmas open on one word the longer wins. A phrasal verb
        written apart holds the words between, so a range may hold another.

        Args:
            tokens: The words of the context, in order.
            lexicon: The lemmas to look for, gathered beforehand.
            text: The context, where the caller handed over one.

        Returns:
            The occurrences, leftmost first and one to a word.
        """
        # A run of whitespace is a word to spaCy and to nobody else, and it
        # falls between the words of a lemma written over two lines.
        word_indices = [
            index for index, token in enumerate(tokens) if not token.form.isspace()
        ]

        word_lemmas = [normalise(tokens[index].lemma) for index in word_indices]

        # A query naming no lemma reads no words, and a span of no words
        # carries no tag to narrow it by.
        word_counts = sorted({len(lemma) for lemma in lexicon if lemma}, reverse=True)

        particles: dict[int, list[int]] = {}

        for index, token in enumerate(tokens):
            if token.particle_of is not None:
                particles.setdefault(token.particle_of, []).append(index)

        matches: list[Match] = []
        cursor = 0

        while cursor < len(word_indices):
            for count in word_counts:
                if cursor + count > len(word_indices):
                    continue

                tags = lexicon.get(tuple(word_lemmas[cursor : cursor + count]))

                if tags is not None and tokens[word_indices[cursor]].pos in tags:
                    span = [
                        tokens[index] for index in word_indices[cursor : cursor + count]
                    ]

                    matches.append(cls._match(span, word_indices[cursor], text))
                    cursor += count

                    break
            else:
                phrasal_verb = cls._phrasal_verb(
                    tokens,
                    word_indices[cursor],
                    word_lemmas[cursor],
                    particles.get(word_indices[cursor], ()),
                    lexicon,
                )

                if phrasal_verb is not None:
                    matches.append(cls._match(phrasal_verb, word_indices[cursor], text))

                cursor += 1

        return tuple(matches)

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
                tokens,
                self._gather_queries(frozenset(queries)),
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
