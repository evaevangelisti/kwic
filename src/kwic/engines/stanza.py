"""
Analysis by Stanza, whose lemmatiser is a dictionary with a model behind it.
"""

from collections.abc import Iterable, Iterator, Sequence
from functools import cached_property
from itertools import batched, groupby
from typing import Protocol, cast, override

from stanza import Document, Pipeline  # pyright: ignore[reportMissingTypeStubs]

from ..constants import BATCH_SIZE, STANZA_LANGUAGE
from ..models import POS, Context, Token
from .base import Engine

PROCESSORS = "tokenize,pos,lemma"
"""What a search reads. Multi-word tokens are expanded where a language has
them."""

DEPPARSE = "depparse"
"""What finds a phrasal verb written apart. Stanza parses with a model of its
own, which doubles what a reading costs."""

PARTICLE = "compound:prt"
"""What Universal Dependencies calls the particle of a phrasal verb."""

LOGGING_LEVEL = "WARN"
"""How much of its loading Stanza reports on the way."""


class _Word(Protocol):
    text: str
    lemma: str | None
    upos: str
    deprel: str | None
    head: int


class _Token(Protocol):
    start_char: int | None
    end_char: int | None
    words: Sequence[_Word]


class _Sentence(Protocol):
    tokens: Sequence[_Token]


class _Document(Protocol):
    sentences: Sequence[_Sentence]


class StanzaEngine(Engine):
    """
    A search reading with Stanza, whose models it downloads on first use.

    A pipeline either tokenises or is handed words already split, so one is
    loaded for each kind of context, and only the kinds asked for.
    """

    def __init__(
        self,
        language: str = STANZA_LANGUAGE,
        batch_size: int = BATCH_SIZE,
        *,
        parse: bool = True,
        gpu: bool = False,
    ) -> None:
        """
        Take the language to read in, loading nothing until asked to read.

        Args:
            language: The language to load the processors for.
            batch_size: How many contexts are read at a time.
            parse: Whether to load the parser, which finds a phrasal verb
            written apart and doubles what a reading costs.
            gpu: Whether to read on the graphics card. Stanza would take one
            wherever it found it; here it is asked for or it is not.
        """
        self._language: str = language

        self._batch_size: int = batch_size
        self._processors: str = f"{PROCESSORS},{DEPPARSE}" if parse else PROCESSORS
        self._gpu: bool = gpu

    @staticmethod
    def _build_document(
        context: Context,
    ) -> Document:
        """
        Hand one context over in the shape a pipeline takes it in.

        Args:
            context: The text or the words to read.

        Returns:
            A document carrying the text, or one carrying the words given.
            The second carries no text, and so no index into one.
        """
        if isinstance(context, tuple):
            words = [
                {"id": index, "text": word}
                for index, word in enumerate(context, start=1)
            ]

            return Document([words])

        return Document([], text=context)

    @staticmethod
    def _read_document(
        document: _Document,
    ) -> Iterator[Token]:
        """
        Read one analysed document into words.

        Args:
            document: The document the pipeline handed back.

        Yields:
            One word at a time. A token holding several words, as doesn't
            does, hands each of them the range the whole token occupies.
        """
        word_index = 0

        for sentence in document.sentences:
            # Stanza numbers the words of a sentence from one, and a document
            # may hold several sentences.
            first_word_index = word_index

            for token in sentence.tokens:
                offsets = (
                    None
                    if token.start_char is None or token.end_char is None
                    else (token.start_char, token.end_char)
                )

                for word in token.words:
                    yield Token(
                        # A word is its own lemma until Stanza says otherwise.
                        lemma=word.lemma if word.lemma is not None else word.text,
                        pos=POS(word.upos),
                        form=word.text,
                        offsets=offsets,
                        particle_of=first_word_index + word.head - 1
                        if word.deprel == PARTICLE
                        else None,
                    )

                    word_index += 1

    @cached_property
    def _tokenising_pipeline(
        self,
    ) -> Pipeline:
        """The pipeline reading a context that has yet to be split."""
        return Pipeline(
            lang=self._language,
            processors=self._processors,
            logging_level=LOGGING_LEVEL,
            use_gpu=self._gpu,
        )

    @cached_property
    def _pretokenised_pipeline(
        self,
    ) -> Pipeline:
        """The pipeline reading a context the caller has split already."""
        return Pipeline(
            lang=self._language,
            processors=self._processors,
            logging_level=LOGGING_LEVEL,
            tokenize_pretokenized=True,
            use_gpu=self._gpu,
        )

    @override
    def analyse_all(
        self,
        contexts: Iterable[Context],
    ) -> Iterator[tuple[Token, ...]]:
        """
        Read every context into its words, a batch at a time.

        Args:
            contexts: The texts to read, or the words they were split into.

        Yields:
            The words of one context, in the order the contexts came in.
        """
        # A pipeline tokenises or is handed words and cannot do both, so a run
        # of one kind is read at a time, which leaves the order alone.
        for is_split, grouped_contexts in groupby(
            contexts, key=lambda context: not isinstance(context, str)
        ):
            pipeline = (
                self._pretokenised_pipeline if is_split else self._tokenising_pipeline
            )

            for batch in batched(grouped_contexts, self._batch_size):
                analysed_documents = cast(
                    "Sequence[_Document]",
                    pipeline([self._build_document(context) for context in batch]),
                )

                for document in analysed_documents:
                    yield tuple(self._read_document(document))
