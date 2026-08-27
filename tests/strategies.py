"""
Generators the properties are drawn from.

A generator says what an engine may read rather than what it usually reads:
a property is only worth stating if the reading nobody thought of can
falsify it, so the alphabets reach past ASCII and the lists reach down to
empty.

Words are drawn out of letters alone. A property comparing two spellings of
one lemma rests on folding the case of it, and a form written with an
apostrophe is spelled out where it is the point of the test.
"""

from hypothesis import strategies as st

from kwic import POS, Query, Token

# Letters as far as Latin Extended-B. Python folds their case the way a
# search does, and comparing two spellings of a lemma rests on that.
_LETTERS = st.characters(categories=("Ll", "Lu"), max_codepoint=0x24F)

words = st.text(alphabet=_LETTERS, min_size=1, max_size=8)
"""One written form, whether a lemma or an inflection of one."""

parts_of_speech: st.SearchStrategy[POS] = st.sampled_from(POS)
"""One of the tags a word may carry."""

tokens = st.builds(
    Token,
    form=words,
    lemma=words,
    pos=parts_of_speech,
    offsets=st.none(),
)
"""One word as an engine read it. What it is written as and what it is read
as are drawn apart, an engine being free to make anything of anything."""

readings = st.lists(tokens, max_size=6).map(tuple)
"""What an engine made of one context, down to the empty context."""

queries = st.builds(Query, lemma=words, pos=st.none() | parts_of_speech)
"""One lemma to look for, under a tag or under any."""

query_lists = st.lists(queries, max_size=4)
"""What a search is asked in one pass, down to being asked nothing."""
