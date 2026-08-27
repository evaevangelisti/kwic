"""
Tests for src/kwic/engines/lemminflect.py.
"""

import pytest

from kwic import SpacyEngine
from kwic.engines.lemminflect import LemmInflectEngine

pytestmark = pytest.mark.pipeline


@pytest.fixture(scope="module")
def lexicon() -> LemmInflectEngine:
    """
    Load the pipeline the tags are read from.

    Returns:
        An engine taking its lemmas from the lexicon.
    """
    return LemmInflectEngine()


def _lemmas(
    engine: SpacyEngine,
    context: str,
) -> list[str]:
    """
    Read one context and keep the lemmas alone.

    Args:
        engine: The engine to read with.
        context: The text to read.

    Returns:
        What each word was read as, in the order they are written.
    """
    return [token.lemma for token in engine.analyse(context)]


class TestLemmas:
    """
    Where a reading comes from, tag by tag.
    """

    def test_takes_a_lemma_the_lexicon_holds(
        self,
        lexicon: LemmInflectEngine,
    ) -> None:
        """An irregular plural is in the lexicon, which is what it is for."""
        assert _lemmas(lexicon, "The geese flew.")[1] == "goose"

    def test_leaves_a_tag_the_lexicon_has_none_for_to_the_pipeline(
        self,
        lexicon: LemmInflectEngine,
        engine: SpacyEngine,
    ) -> None:
        """The lexicon holds nothing under AUX, and the rules read is as be."""
        context = "It is over."

        assert _lemmas(lexicon, context)[1] == _lemmas(engine, context)[1] == "be"
