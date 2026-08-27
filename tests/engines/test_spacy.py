"""
Tests for src/kwic/engines/spacy.py.
"""

from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st

from kwic import POS, SpacyEngine

pytestmark = pytest.mark.pipeline

# What an English context is written in, whitespace and the punctuation a
# tokeniser has to decide about included.
_CONTEXTS = st.text(
    alphabet=st.characters(codec="ascii", min_codepoint=32),
    max_size=40,
)

_WORDS = st.lists(
    st.text(alphabet=st.characters(categories=("Ll", "Lu")), min_size=1, max_size=8),
    max_size=6,
).map(tuple)


class TestRanges:
    """
    Where an engine says a word falls.
    """

    @given(context=_CONTEXTS)
    def test_slices_every_word_back_out_of_the_context(
        self,
        context: str,
        engine: SpacyEngine,
    ) -> None:
        """A range is half-open and in code points, as Python indexes a string."""
        for token in engine.analyse(context):
            assert token.offsets is not None
            assert context[token.offsets[0] : token.offsets[1]] == token.form

    @given(context=_CONTEXTS)
    def test_reads_the_words_in_the_order_they_are_written(
        self,
        context: str,
        engine: SpacyEngine,
    ) -> None:
        """A context is read left to right, and no word is read into another."""
        offsets = [token.offsets for token in engine.analyse(context)]

        assert all(
            before is not None and after is not None and before[1] <= after[0]
            for before, after in pairwise(offsets)
        )

    def test_splits_a_contraction_into_words_of_its_own(
        self,
        engine: SpacyEngine,
    ) -> None:
        """Where spaCy splits a token it gives each half a range of its own."""
        does, apostrophe = engine.analyse("doesn't")[:2]

        assert (does.lemma, does.offsets) == ("do", (0, 4))
        assert (apostrophe.lemma, apostrophe.offsets) == ("not", (4, 7))


class TestContexts:
    """
    What an engine takes, and what it hands back for each kind.
    """

    @given(words=_WORDS)
    def test_reads_a_context_split_beforehand_into_the_words_given(
        self,
        words: tuple[str, ...],
        engine: SpacyEngine,
    ) -> None:
        """The words the caller split are the words that come back, and no others."""
        assert tuple(token.form for token in engine.analyse(words)) == words

    @given(words=_WORDS)
    def test_places_a_context_split_beforehand_in_no_text(
        self,
        words: tuple[str, ...],
        engine: SpacyEngine,
    ) -> None:
        """There is no text for a range to point into, so none is reported."""
        assert all(token.offsets is None for token in engine.analyse(words))


class TestParticles:
    """
    Which word a phrasal verb written apart is the particle of.
    """

    def test_marks_the_particle_with_the_verb_it_belongs_to(
        self,
        engine: SpacyEngine,
    ) -> None:
        """Up belongs to gave, three words back, and the parser is what says so."""
        analysed_words = engine.analyse("She gave the money up.")

        assert [token.particle_of for token in analysed_words] == [
            None,
            None,
            None,
            None,
            1,
            None,
        ]

    def test_marks_nothing_where_it_reads_without_the_parser(
        self,
    ) -> None:
        """A pipeline read without the parser labels nothing to belong anywhere."""
        analysed_words = SpacyEngine(parse=False).analyse("She gave the money up.")

        assert all(token.particle_of is None for token in analysed_words)


class TestSplitting:
    """
    How a query is cut into words, a lemma having to be cut as a context is.
    """

    @given(context=_CONTEXTS)
    def test_cuts_a_text_into_the_words_it_is_read_as(
        self,
        context: str,
        engine: SpacyEngine,
    ) -> None:
        """Cutting and reading are the same cut, one of them without the reading."""
        analysed_words = tuple(token.form for token in engine.analyse(context))

        assert next(engine.split_all([context])) == analysed_words

    def test_cuts_a_hyphenated_word_into_the_words_it_holds(
        self,
        engine: SpacyEngine,
    ) -> None:
        """A hyphen is a word to spaCy, whatever a space would say."""
        assert next(engine.split_all(["pre-university"])) == ("pre", "-", "university")


class TestTags:
    """
    Which tagset a reading is reported in.
    """

    @given(context=_CONTEXTS)
    def test_tags_every_word_with_a_universal_tag(
        self,
        context: str,
        engine: SpacyEngine,
    ) -> None:
        """Whitespace is tagged with what UD keeps for what it has no name for."""
        assert all(token.pos in POS for token in engine.analyse(context))

    def test_reads_a_run_of_whitespace_as_a_word_of_its_own(
        self,
        engine: SpacyEngine,
    ) -> None:
        """A context is written back out as it came, so the run stands among them."""
        analysed_words = engine.analyse("a  b")

        assert (analysed_words[1].form, analysed_words[1].pos) == (" ", POS.X)
