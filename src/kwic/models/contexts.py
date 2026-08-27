"""
What a search is run over.
"""

type Context = str | tuple[str, ...]
"""One context: a text, or the words a caller split it into beforehand. Words
split beforehand leave an occurrence with no text to be placed in."""
