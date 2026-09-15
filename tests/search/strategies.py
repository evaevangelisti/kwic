"""
Generators for search properties.

They include unusual readings, Unicode letters, and empty collections.
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
