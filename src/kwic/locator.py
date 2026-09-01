"""
The search a caller runs.
"""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import tee

from .engines import Engine, SpacyEngine
from .models import POS, Context, Match, Query, Token
from .normalisation import normalise, spellings


@dataclass(frozen=True, slots=True)
class Lexicon:
    """
    What a search looks a word up in, everything in it normalised.

    Attributes:
        lemmas: The tags each queried lemma is read under, cut into words.
        forms: The tags each queried form is read under, cut the same way.
    """

    lemmas: Mapping[tuple[str, ...], frozenset[POS]]
    forms: Mapping[tuple[str, ...], frozenset[POS]]


ALL_TAGS = frozenset(POS)
"""What a query naming no part of speech narrows to, which is no narrowing."""

PARTICLE_WORDS = 2
"""How many words a phrasal verb is, its particle written apart or not."""

PARAGRAPH = 2
"""How many line breaks part two blocks rather than wrap one line."""


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
        self._last_lexicon: Lexicon = Lexicon(lemmas={}, forms={})

    def _keys(
        self,
        spelled: Sequence[tuple[Query, str]],
    ) -> dict[tuple[str, ...], frozenset[POS]]:
        """
        Cut every spelling into the words a context would be read as.

        Args:
            spelled: A query and one way of writing what it looks for.

        Returns:
            The tags to look each cut up under, the whole spelling standing
            beside the cut for a text that writes it as one word.
        """
        keys: dict[tuple[str, ...], frozenset[POS]] = {}
        cuts = self._engine.split_all(spelling for _, spelling in spelled)

        for (query, spelling), words in zip(spelled, cuts, strict=True):
            tags = ALL_TAGS if query.pos is None else frozenset({query.pos})
            cut = tuple(normalise(word) for word in words if not word.isspace())

            for key in (cut, (normalise(spelling),)):
                keys[key] = keys.get(key, frozenset()) | tags

        return keys

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
            The lemmas and the forms to look for, every tag standing in where
            a query named none.
        """
        if queries == self._last_queries:
            return self._last_lexicon

        ordered_queries = list(queries)

        self._last_queries = queries
        self._last_lexicon = Lexicon(
            lemmas=self._keys(
                [
                    (query, spelling)
                    for query in ordered_queries
                    for spelling in spellings(query.lemma)
                ]
            ),
            forms=self._keys(
                [
                    (query, spelling)
                    for query in ordered_queries
                    for form in query.forms
                    for spelling in spellings(form)
                ]
            ),
        )

        return self._last_lexicon

    @staticmethod
    def _span_tags(
        lexicon: Lexicon,
        word_lemmas: Sequence[str],
        word_forms: Sequence[str],
        cursor: int,
        count: int,
    ) -> frozenset[POS] | None:
        """
        Read the tags a span of words is looked for under.

        Args:
            lexicon: The lemmas and the forms to look for.
            word_lemmas: What each word was read as.
            word_forms: How each word is written.
            cursor: Where the span opens.
            count: How many words it holds.

        Returns:
            The tags, or None where nothing looks for that span.
        """
        tags = lexicon.lemmas.get(tuple(word_lemmas[cursor : cursor + count]))

        # A form stands in where the engine read a word as another lemma.
        if tags is None:
            return lexicon.forms.get(tuple(word_forms[cursor : cursor + count]))

        return tags

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
            if (lemma, normalise(tokens[particle].lemma)) in lexicon.lemmas:
                return (tokens[index], tokens[particle])

        return None

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
        # A run of whitespace is a word to spaCy and to nobody else, though a
        # blank line parts two blocks and holds the words of a lemma apart.
        word_indices = [
            index
            for index, token in enumerate(tokens)
            if not token.form.isspace() or token.form.count("\n") >= PARAGRAPH
        ]

        word_lemmas = [normalise(tokens[index].lemma) for index in word_indices]
        word_forms = [normalise(tokens[index].form) for index in word_indices]

        # A query naming no lemma reads no words, and a span of no words
        # carries no tag to narrow it by.
        word_counts = sorted(
            {len(key) for key in (*lexicon.lemmas, *lexicon.forms) if key},
            reverse=True,
        )

        particles: dict[int, list[int]] = {}

        for index, token in enumerate(tokens):
            if token.particle_of is not None:
                particles.setdefault(token.particle_of, []).append(index)

        matches: list[Match] = []
        cursor = 0

        while cursor < len(word_indices):
            word_index = word_indices[cursor]

            span: Sequence[Token] | None = None
            step = 1

            for count in word_counts:
                if cursor + count <= len(word_indices):
                    inside = word_indices[cursor : cursor + count]

                    tags = cls._span_tags(
                        lexicon, word_lemmas, word_forms, cursor, count
                    )

                    # A lemma of several words is an expression of its
                    # own, whose category none of its words need carry.
                    if tags is not None and (
                        count > 1 or tokens[word_index].pos in tags
                    ):
                        span = [tokens[index] for index in inside]
                        step = count

                        break

                # A phrasal verb is two words wherever its particle sits, so
                # it is read before the verb alone is read as a lemma.
                if count == PARTICLE_WORDS:
                    span = cls._phrasal_verb(
                        tokens,
                        word_index,
                        word_lemmas[cursor],
                        particles.get(word_index, ()),
                        lexicon,
                    )

                    if span is not None:
                        break

            if span is not None:
                matches.append(cls._match(span, word_index, text))

            cursor += step

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
