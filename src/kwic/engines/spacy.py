"""
Analysis by spaCy, which is what a search reads with unless told otherwise.
"""

from collections.abc import Iterable, Iterator
from typing import override

import spacy
from spacy.language import Language
from spacy.tokens import Doc
from spacy.tokens import Token as Word

from ..constants import BATCH_SIZE, PROCESSES, SPACY_PIPELINE
from ..models import POS, Context, Token
from .base import Engine

EXCLUDED = ("senter", "ner")
"""The components a search never reads, left unloaded rather than turned off.
What assigns a tag stays, the English lemmatiser being rules that read one."""

PARSER = "parser"
"""What finds a phrasal verb written apart, and what several of the rules
assigning a universal tag read. It is worth three points of F1."""

PARTICLE = "prt"
"""What spaCy calls the particle of a phrasal verb."""

WHITESPACE = "SPACE"
"""What spaCy tags a run of whitespace, UD having no tag for one."""


class SpacyEngine(Engine):
    """
    A search reading with a spaCy pipeline.

    English lemmatisation in spaCy is rule-based, the tagger deciding and the
    lemmatiser applying the rules that tag calls for.
    """

    def __init__(
        self,
        pipeline: str = SPACY_PIPELINE,
        batch_size: int = BATCH_SIZE,
        processes: int = PROCESSES,
        *,
        parse: bool = True,
        gpu: bool = False,
    ) -> None:
        """
        Load a pipeline, holding on to it for every context to come.

        Args:
            pipeline: The installed pipeline to load, such as en_core_web_sm.
            batch_size: How many contexts it reads at a time.
            processes: How many processes it may run. One is what a card
            wants, several readings of one card contending rather than
            helping.
            parse: Whether to load the parser, which finds a phrasal verb
            written apart and settles several of the universal tags.
            gpu: Whether to read on the graphics card, which a transformer
            is worth asking for. Raises rather than falling back, a reading
            that quietly went to the processor being one nobody notices.

        Raises:
            OSError: If the pipeline is not installed.
            ValueError: If the graphics card was asked for and none answers.
            ValueError: When read, if a context is longer than spaCy's
            nlp.max_length, a million characters.
        """
        if gpu:
            _ = spacy.require_gpu()  # pyright: ignore[reportPrivateImportUsage]

        self._pipeline: Language = spacy.load(
            pipeline,
            exclude=list(EXCLUDED if parse else (PARSER, *EXCLUDED)),
        )

        self._batch_size: int = batch_size
        self._processes: int = processes

    @staticmethod
    def _tag(
        pos: str,
    ) -> POS:
        """
        Read one spaCy tag as the universal tag it stands for.

        Args:
            pos: The coarse tag spaCy assigned.

        Returns:
            The universal tag, whitespace falling under X.
        """
        return POS.X if pos == WHITESPACE else POS(pos)

    def _lemma(
        self,
        word: Word,
    ) -> str:
        """
        Read the lemma of one word, which is where an engine may differ.

        Args:
            word: The word the pipeline read.

        Returns:
            What the rules of the pipeline made of it.
        """
        return word.lemma_

    def _prepare_context(
        self,
        context: Context,
    ) -> str | Doc:
        """
        Hand one context over in the shape the pipeline takes it in.

        Args:
            context: The text or the words to read.

        Returns:
            The text as it stands, or a document holding the words given.
        """
        if isinstance(context, tuple):
            return Doc(self._pipeline.vocab, words=list(context))

        return context

    def _read_document(
        self,
        document: Doc,
    ) -> Iterator[Token]:
        """
        Read one analysed document into words.

        Args:
            document: The document the pipeline handed back.

        Yields:
            One word at a time, in the order they are written.
        """
        # A document built from words alone stands in a text nobody handed
        # over, and spaCy indexes it as though the words were spaced out.
        is_located = not document.has_unknown_spaces

        for word in document:
            yield Token(
                lemma=self._lemma(word),
                pos=self._tag(word.pos_),
                form=word.text,
                offsets=(word.idx, word.idx + len(word)) if is_located else None,
                # Read without the parser, every word is labelled with nothing.
                particle_of=word.head.i if word.dep_ == PARTICLE else None,
            )

    @override
    def split_all(
        self,
        texts: Iterable[str],
    ) -> Iterator[tuple[str, ...]]:
        """
        Cut every text into words, without reading any of them.

        Args:
            texts: The texts to cut.

        Yields:
            The words of one text, in the order they are written.
        """
        for text in texts:
            yield tuple(word.text for word in self._pipeline.tokenizer(text))

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
        prepared_contexts = (self._prepare_context(context) for context in contexts)

        for document in self._pipeline.pipe(
            prepared_contexts,
            batch_size=self._batch_size,
            n_process=self._processes,
        ):
            yield tuple(self._read_document(document))
