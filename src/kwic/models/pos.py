"""
The parts of speech a query may name.
"""

from enum import StrEnum


class POS(StrEnum):
    """
    The universal part-of-speech tags, as Universal Dependencies writes them.

    Values are UD's own codes, so POS("VERB") converts directly, and every
    engine reports them whatever tagset its model was trained on.
    """

    ADJ = "ADJ"
    ADP = "ADP"
    ADV = "ADV"
    AUX = "AUX"
    CCONJ = "CCONJ"
    DET = "DET"
    INTJ = "INTJ"
    NOUN = "NOUN"
    NUM = "NUM"
    PART = "PART"
    PRON = "PRON"
    PROPN = "PROPN"
    PUNCT = "PUNCT"
    SCONJ = "SCONJ"
    SYM = "SYM"
    VERB = "VERB"
    X = "X"
