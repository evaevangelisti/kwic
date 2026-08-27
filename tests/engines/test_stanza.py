"""
Tests for src/kwic/engines/stanza.py.
"""

import pytest

from kwic.engines.stanza import StanzaEngine

pytestmark = pytest.mark.pipeline

_CONTRACTION = "She doesn't know."


@pytest.fixture(scope="module")
def stanza() -> StanzaEngine:
    """
    Take an engine, whose models are loaded when it is first read with.

    Returns:
        An engine reading English.
    """
    return StanzaEngine()


class TestRanges:
    """
    Where an engine says a word falls.
    """

    def test_slices_a_word_back_out_of_the_context(
        self,
        stanza: StanzaEngine,
    ) -> None:
        """A range is half-open and in code points, as Python indexes a string."""
        know = stanza.analyse(_CONTRACTION)[3]

        assert know.offsets is not None
        assert _CONTRACTION[know.offsets[0] : know.offsets[1]] == know.form == "know"

    def test_gives_every_word_of_a_token_the_range_of_the_token(
        self,
        stanza: StanzaEngine,
    ) -> None:
        """Does and n't are written nowhere but inside doesn't, which is the range."""
        does, apostrophe = stanza.analyse(_CONTRACTION)[1:3]

        assert (does.form, does.offsets) == ("does", (4, 11))
        assert (apostrophe.form, apostrophe.offsets) == ("n't", (4, 11))


class TestContexts:
    """
    What an engine takes, and what it hands back for each kind.
    """

    def test_reads_a_context_split_beforehand_into_the_words_given(
        self,
        stanza: StanzaEngine,
    ) -> None:
        """The words the caller split are the words that come back, in no text."""
        analysed_words = stanza.analyse(("The", "leaves", "fell"))

        assert [(token.form, token.offsets) for token in analysed_words] == [
            ("The", None),
            ("leaves", None),
            ("fell", None),
        ]
