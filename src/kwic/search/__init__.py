"""
Reading the queried lemmas off a context.

A lexicon is what a word is looked up in, a reading is the context laid
out to be looked up, and a locator is what a caller runs.
"""

from .locator import Locator

__all__ = [
    "Locator",
]
