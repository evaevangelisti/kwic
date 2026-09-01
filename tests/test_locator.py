"""
Tests for src/kwic/locator.py.
"""

import string
from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st
from strategies import parts_of_speech, queries, query_lists, readings, words
from stubs import Given

from kwic import POS, Locator, Match, Query, Token

_CONTEXT = "The leaves fell as she leaves the room."

# ASCII alone, since only there is raising a letter the undoing of folding
# it: the dotless i of Turkish is raised to I and folded to itself.
_ASCII_QUERIES = st.lists(
    st.builds(
        Query,
        lemma=st.text(alphabet=string.ascii_letters, min_size=1, max_size=6),
        pos=st.none() | parts_of_speech,
    ),
    max_size=4,
)


def _asks_for(
    query: Query,
    token: Token,
) -> bool:
    """
    Say whether one query asks for one word.

    Args:
        query: The lemma looked for, under a tag or under any.
        token: The word as the engine read it.

    Returns:
        Whether they agree on the lemma, and on the tag where one is named.
    """
    return query.lemma.casefold() == token.lemma.casefold() and query.pos in (
        None,
        token.pos,
    )


def _find_matches(
    reading: tuple[Token, ...],
    questions: list[Query],
) -> tuple[Match, ...]:
    """
    Put queries to a search whose engine has read the context already.

    Args:
        reading: What the engine made of it.
        questions: The lemmas to look for.

    Returns:
        The occurrences the search hands back.
    """
    context = " ".join(token.form for token in reading)

    return Locator(Given(reading)).find(context, questions)


class TestOccurrences:
    """
    What comes back for one context.
    """

    @given(readings, query_lists)
    def test_hands_back_the_word_the_engine_read(
        self,
        reading: tuple[Token, ...],
        questions: list[Query],
    ) -> None:
        """Everything but the lemma is reported as the engine read it."""
        assert all(
            (match.pos, match.form, match.offsets)
            == (
                reading[match.word_index].pos,
                reading[match.word_index].form,
                reading[match.word_index].offsets,
            )
            for match in _find_matches(reading, questions)
        )

    def test_hands_back_the_lemma_it_was_asked_for(
        self,
    ) -> None:
        """The lemma is the caller's own, not what the engine made of the word."""
        reading = (Token(lemma="thank", pos=POS.NOUN, form="Thanks", offsets=None),)
        asked = Query("thanks", POS.NOUN, frozenset({"Thanks"}))

        assert _find_matches(reading, [asked])[0].lemma == "thanks"

    @given(readings, query_lists)
    def test_reads_every_word_a_query_asks_for(
        self,
        reading: tuple[Token, ...],
        questions: list[Query],
    ) -> None:
        """A context attesting the lemma five times is evidence five times over."""
        expected = tuple(
            index
            for index, token in enumerate(reading)
            if any(_asks_for(query, token) for query in questions)
        )

        assert (
            tuple(match.word_index for match in _find_matches(reading, questions))
            == expected
        )

    @given(readings, query_lists)
    def test_reads_leftmost_first_and_never_twice_over(
        self,
        reading: tuple[Token, ...],
        questions: list[Query],
    ) -> None:
        """A word is one occurrence, whichever of the queries reached it."""
        found = _find_matches(reading, questions)

        assert all(
            before.word_index < after.word_index for before, after in pairwise(found)
        )

    @given(readings)
    def test_hands_back_nothing_when_nothing_is_asked(
        self,
        reading: tuple[Token, ...],
    ) -> None:
        """A search asked for no lemma has no lemma to look for."""
        assert _find_matches(reading, []) == ()

    @given(st.data())
    def test_hands_back_nothing_when_the_lemma_is_absent(
        self,
        data: st.DataObject,
    ) -> None:
        """A context need not attest what it is read for."""
        reading = data.draw(readings)
        read_lemmas = {token.lemma.casefold() for token in reading}
        lemma = data.draw(words.filter(lambda word: word.casefold() not in read_lemmas))

        assert _find_matches(reading, [Query(lemma)]) == ()


class TestQueries:
    """
    Which lemmas a search looks for, and how it is told to.
    """

    @given(readings, _ASCII_QUERIES)
    def test_takes_a_lemma_in_any_case(
        self,
        reading: tuple[Token, ...],
        questions: list[Query],
    ) -> None:
        """A lemma opening a sentence is capitalised, and it is the lemma."""
        raised = [Query(query.lemma.upper(), query.pos) for query in questions]

        assert [match.word_index for match in _find_matches(reading, raised)] == [
            match.word_index for match in _find_matches(reading, questions)
        ]

    @given(readings, queries)
    def test_reads_a_lemma_asked_for_twice_once(
        self,
        reading: tuple[Token, ...],
        query: Query,
    ) -> None:
        """Two queries reaching one word are one occurrence of it."""
        assert _find_matches(reading, [query, query]) == _find_matches(reading, [query])

    @given(readings, query_lists, queries)
    def test_asking_for_more_finds_no_less(
        self,
        reading: tuple[Token, ...],
        questions: list[Query],
        query: Query,
    ) -> None:
        """A lemma added to a search takes nothing away from the ones there."""
        found = {match.word_index for match in _find_matches(reading, questions)}
        widened = {
            match.word_index for match in _find_matches(reading, [*questions, query])
        }

        assert found <= widened

    @given(readings, words)
    def test_a_query_naming_no_tag_takes_every_reading_of_the_lemma(
        self,
        reading: tuple[Token, ...],
        lemma: str,
    ) -> None:
        """Naming a tag narrows a search, and naming none is the whole of it."""
        under_any = _find_matches(reading, [Query(lemma)])
        under_each = _find_matches(reading, [Query(lemma, pos) for pos in POS])

        assert under_any == under_each

    def test_takes_a_word_written_as_a_form_it_was_given(
        self,
    ) -> None:
        """A form stands in where the engine read the word as another lemma."""
        reading = (Token(lemma="thank", pos=POS.NOUN, form="Thanks", offsets=(0, 6)),)

        assert _find_matches(reading, [Query("thanks", POS.NOUN)]) == ()
        assert _find_matches(
            reading, [Query("thanks", POS.NOUN, frozenset({"Thanks"}))]
        )

    def test_takes_a_lemma_however_its_apostrophe_is_set(
        self,
    ) -> None:
        """A context set typographically is read for a lemma typed straight."""
        reading = (Token(lemma="don’t", pos=POS.VERB, form="don’t", offsets=(0, 5)),)

        assert _find_matches(reading, [Query("don't")])[0].form == "don’t"


class TestLemmasOfSeveralWords:
    """
    What a search makes of a lemma written as more than one word.
    """

    @given(st.data())
    def test_reads_it_as_its_words_in_order(
        self,
        data: st.DataObject,
    ) -> None:
        """Give up is looked for where give and up stand one after the other."""
        reading = data.draw(readings.filter(lambda drawn: len(drawn) >= 2))
        index = data.draw(st.integers(min_value=0, max_value=len(reading) - 2))
        span = reading[index : index + 2]

        lemma = " ".join(token.lemma for token in span)
        found = _find_matches(reading, [Query(lemma)])

        assert found
        assert all(match.lemma.casefold() == lemma.casefold() for match in found)

    def test_takes_a_form_of_several_words(
        self,
    ) -> None:
        """A form is cut like a lemma, so one of several words is looked for."""
        reading = (
            Token(lemma="electric", pos=POS.ADJ, form="electric", offsets=None),
            Token(lemma="eels", pos=POS.NOUN, form="eels", offsets=None),
        )

        assert _find_matches(reading, [Query("electric eel")]) == ()
        assert _find_matches(
            reading, [Query("electric eel", None, frozenset({"electric eels"}))]
        )

    def test_takes_the_longest_lemma_that_fits(
        self,
    ) -> None:
        """A phrasal verb is what it is, not the verb that opens it."""
        reading = (
            Token(lemma="give", pos=POS.VERB, form="gave", offsets=None),
            Token(lemma="up", pos=POS.ADP, form="up", offsets=None),
        )

        found = _find_matches(reading, [Query("give"), Query("give up")])

        assert [(match.lemma, match.form) for match in found] == [
            ("give up", "gave up")
        ]

    def test_is_not_narrowed_by_a_tag_at_all(
        self,
    ) -> None:
        """A dictionary tags the whole expression, which no one word need carry."""
        reading = (
            Token(lemma="electric", pos=POS.ADJ, form="electric", offsets=None),
            Token(lemma="eel", pos=POS.NOUN, form="eel", offsets=None),
        )

        assert all(_find_matches(reading, [Query("electric eel", pos)]) for pos in POS)

    def test_stops_at_a_blank_line_between_its_words(
        self,
    ) -> None:
        """A blank line parts two blocks, and no lemma runs across one."""
        reading = (
            Token(lemma="give", pos=POS.VERB, form="give", offsets=(0, 4)),
            Token(lemma="\n\n", pos=POS.X, form="\n\n", offsets=(4, 6)),
            Token(lemma="up", pos=POS.ADP, form="up", offsets=(6, 8)),
        )

        assert _find_matches(reading, [Query("give up")]) == ()

    def test_steps_over_the_whitespace_between_its_words(
        self,
    ) -> None:
        """A lemma written over two lines is the lemma, and spaCy reads the break."""
        reading = (
            Token(lemma="give", pos=POS.VERB, form="gave", offsets=(0, 4)),
            Token(lemma="\n", pos=POS.X, form="\n", offsets=(4, 5)),
            Token(lemma="up", pos=POS.ADP, form="up", offsets=(5, 7)),
        )

        found = Locator(Given(reading)).find("gave\nup", [Query("give up")])

        assert [(match.form, match.offsets) for match in found] == [
            ("gave\nup", (0, 7))
        ]


class TestPhrasalVerbsWrittenApart:
    """
    What a search makes of a verb and a particle with words between them.
    """

    _READING: tuple[Token, ...] = (
        Token(lemma="give", pos=POS.VERB, form="gave", offsets=(0, 4)),
        Token(lemma="it", pos=POS.PRON, form="it", offsets=(5, 7)),
        Token(lemma="up", pos=POS.ADP, form="up", offsets=(8, 10), particle_of=0),
    )

    _CONTEXT: str = "gave it up"

    def test_reads_the_verb_and_the_particle_as_one_lemma(
        self,
    ) -> None:
        """Give up is give up, whatever was put between the two."""
        found = Locator(Given(self._READING)).find(self._CONTEXT, [Query("give up")])

        assert [(match.lemma, match.form, match.offsets) for match in found] == [
            ("give up", "gave it up", (0, 10))
        ]

    def test_leaves_the_two_apart_where_the_parser_did_not_join_them(
        self,
    ) -> None:
        """Without a parser no word is the particle of another, and up is a word."""
        reading = tuple(
            Token(
                form=token.form, lemma=token.lemma, pos=token.pos, offsets=token.offsets
            )
            for token in self._READING
        )

        assert Locator(Given(reading)).find(self._CONTEXT, [Query("give up")]) == ()

    def test_wins_over_the_verb_it_opens_on(
        self,
    ) -> None:
        """The longer lemma wins, the particle written apart or beside it."""
        found = Locator(Given(self._READING)).find(
            self._CONTEXT, [Query("give"), Query("give up")]
        )

        assert [(match.lemma, match.form) for match in found] == [
            ("give up", "gave it up")
        ]

    def test_reads_the_words_between_for_what_else_is_asked(
        self,
    ) -> None:
        """A word inside the range is a word still, and one range may hold another."""
        found = Locator(Given(self._READING)).find(
            self._CONTEXT, [Query("give up"), Query("it")]
        )

        assert [(match.lemma, match.offsets) for match in found] == [
            ("give up", (0, 10)),
            ("it", (5, 7)),
        ]


class TestManyContexts:
    """
    What a search makes of the contexts of a run.
    """

    @given(readings, st.lists(query_lists, max_size=4))
    def test_answers_the_searches_in_the_order_they_came(
        self,
        reading: tuple[Token, ...],
        questions: list[list[Query]],
    ) -> None:
        """A run of searches is the searches run one at a time, in their order."""
        locator = Locator(Given(reading))
        context = " ".join(token.form for token in reading)

        assert list(
            locator.find_all([(context, queries) for queries in questions])
        ) == [locator.find(context, queries) for queries in questions]

    def test_reads_each_context_for_the_lemmas_asked_of_it(
        self,
    ) -> None:
        """A search is a context and the lemmas meant for that context."""
        reading = (Token(lemma="leaf", pos=POS.NOUN, form="leaves", offsets=(0, 6)),)

        found = Locator(Given(reading)).find_all(
            [("leaves", [Query("leaf")]), ("leaves", [Query("leave")])]
        )

        assert [len(matches) for matches in found] == [1, 0]


@pytest.mark.pipeline
class TestReadings:
    """
    What a search makes of a context a model has read.
    """

    def test_tells_two_readings_of_one_form_apart(
        self,
        locator: Locator,
    ) -> None:
        """The leaves that fell are not the leaves she left, spelled alike or not."""
        found = locator.find(_CONTEXT, [Query("leaf", POS.NOUN)])

        assert [match.offsets for match in found] == [(4, 10)]

    def test_takes_a_lemma_and_leaves_the_form_of_another(
        self,
        locator: Locator,
    ) -> None:
        """Found is an occurrence of find, and found is a lemma of its own."""
        context = "She found the keys she had lost."

        assert locator.find(context, [Query("find", POS.VERB)])
        assert locator.find(context, [Query("found", POS.VERB)]) == ()

    def test_slices_the_form_back_out_of_the_context(
        self,
        locator: Locator,
    ) -> None:
        """A range is half-open and in code points, as Python indexes a string."""
        found = locator.find(_CONTEXT, [Query("leaf"), Query("leave")])

        assert all(
            match.offsets is not None
            and _CONTEXT[match.offsets[0] : match.offsets[1]] == match.form
            for match in found
        )

    def test_reads_a_phrasal_verb_written_apart(
        self,
        locator: Locator,
    ) -> None:
        """The parser says which up belongs to which verb, however far it sits."""
        context = "She gave the money up."
        found = locator.find(context, [Query("give up", POS.VERB)])

        assert [(match.form, match.offsets) for match in found] == [
            ("gave the money up", (4, 21))
        ]

    def test_takes_a_lemma_however_its_words_are_parted(
        self,
        locator: Locator,
    ) -> None:
        """A dictionary writes hunky dory where a text writes hunky-dory."""
        assert locator.find("It was hunky-dory.", [Query("hunky dory")])
        assert locator.find("It was hunky dory.", [Query("hunky-dory")])
        assert locator.find("A back-seat driver.", [Query("back seat driver")])

    def test_places_a_context_split_beforehand_by_its_words(
        self,
        locator: Locator,
    ) -> None:
        """There is no text for a range to point into, so the word is placed."""
        found = locator.find(("She", "found", "the", "keys"), [Query("find")])

        assert [(match.word_index, match.offsets) for match in found] == [(1, None)]

    def test_cuts_a_lemma_the_way_it_cuts_a_context(
        self,
        locator: Locator,
    ) -> None:
        """Pre-university is one word to a space and three to the tokeniser."""
        context = "She took a pre-university course."
        found = locator.find(context, [Query("pre-university")])

        assert [(match.form, match.offsets) for match in found] == [
            ("pre-university", (11, 25))
        ]

    def test_reads_a_lemma_of_several_words(
        self,
        locator: Locator,
    ) -> None:
        """A range opens on the first word of the lemma and closes on the last."""
        context = "She gave up on it."
        found = locator.find(context, [Query("give up", POS.VERB)])

        assert [(match.form, match.offsets) for match in found] == [
            ("gave up", (4, 11))
        ]
