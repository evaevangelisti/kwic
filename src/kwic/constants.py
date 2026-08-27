"""
Values a search never varies.
"""

SPACY_PIPELINE = "en_core_web_sm"
"""The spaCy pipeline loaded unless another is named. What the larger English
pipelines add is vectors, which a search does not read."""

STANZA_LANGUAGE = "en"
"""The language Stanza loads its processors for unless another is named."""

BATCH_SIZE = 256
"""How many contexts a model reads at a time. Larger batches amortise the call
into the model and hold more contexts in memory."""

PROCESSES = 1
"""How many processes a pipeline may run. More than one pays off over long
runs alone, each having to load the model again."""
