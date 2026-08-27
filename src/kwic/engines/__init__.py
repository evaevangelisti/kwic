"""
The analysers a search may read with.

Only spaCy is loaded from here, it being the one dependency the package
takes. The others are extras, imported from the module that wraps them.
"""

from .base import Engine
from .spacy import SpacyEngine

__all__ = [
    "Engine",
    "SpacyEngine",
]
