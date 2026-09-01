"""
Measuring the engines against a treebank.

The study puts the same queries to every engine and counts what each returns
against what the treebank records. The queries are drawn from the treebank
itself, and put twice: under the tag it gives the lemma, and under any tag.

An engine reads every sentence once and is asked both sets of queries against
that reading, the reading being what a run spends its time on.

What the figures do not settle. Stanza is trained on a bundle both sections
belong to and spaCy on OntoNotes, so a treebank scores Stanza at home. A
lemma is a convention as much as a fact, and the two engines learned
different ones. An occurrence counts as found when its range is the range the
treebank gives, so an engine cutting a token finer is marked wrong for it.

The tables are written as Markdown, one file per pass, named for the moment.
"""

import argparse
import random
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import cast, override

import treebanks
from lemminflect import (  # pyright: ignore[reportMissingTypeStubs]
    getAllInflections,  # pyright: ignore[reportUnknownVariableType]
    getAllInflectionsOOV,  # pyright: ignore[reportUnknownVariableType]
)
from treebanks import Sentence, Word

from kwic import POS, Context, Engine, Locator, Query, SpacyEngine, Token
from kwic.engines.lemminflect import LemmInflectEngine
from kwic.engines.stanza import StanzaEngine
from kwic.normalisation import normalise

EVALUATIONS = Path(__file__).resolve().parent / "evaluations"
"""Where the written evaluations are kept, one file per pass."""

STAMP = "%Y%m%dT%H%M%SZ"
"""How an evaluation names itself, the moment telling two of them apart."""

ENGINES: dict[str, Callable[[], Engine]] = {
    "spacy-sm": lambda: SpacyEngine("en_core_web_sm"),
    "spacy-lg": lambda: SpacyEngine("en_core_web_lg"),
    "spacy-trf": lambda: SpacyEngine("en_core_web_trf"),
    "lemminflect": lambda: LemmInflectEngine("en_core_web_sm"),
    "stanza": StanzaEngine,
}
"""The engines the study runs, each loaded only if it is asked for."""

QUERIED = (POS.ADJ, POS.ADV, POS.NOUN, POS.PROPN, POS.VERB)
"""The tags a query is drawn under. Punctuation and function words are what
every engine agrees on, and counting them would report the agreement."""

QUERIES = 500
"""How many lemmas are drawn unless another number is given."""

SEED = 0
"""What the sample is drawn with, so that two runs ask the same questions."""

OWN_TAG = "Under their own tag"
"""What the queries drawn from the treebank are asked under."""

ANY_TAG = "Under any tag"
"""What the same lemmas are asked under, their tag dropped."""

type Occurrence = tuple[int, int, int]
"""One occurrence: the sentence it falls in, and the range it occupies."""


def say(
    message: str,
) -> None:
    """
    Report how far along a run is, while it is running.

    Args:
        message: What to say.
    """
    print(message, flush=True)  # noqa: T201


class Remembered(Engine):
    """
    An engine reading each context once and remembering what it read.

    The study asks two sets of queries of one reading. Reading is what costs,
    and a reading is the same whatever it is asked.
    """

    def __init__(
        self,
        engine: Engine,
    ) -> None:
        """
        Take the engine that does the reading.

        Args:
            engine: The engine to read with.
        """
        self._engine: Engine = engine
        self._readings: dict[Context, tuple[Token, ...]] = {}

    def remember(
        self,
        contexts: Sequence[Context],
    ) -> None:
        """
        Read a batch of contexts, holding on to what was read of each.

        Args:
            contexts: The texts to read.
        """
        self._readings.update(
            zip(contexts, self._engine.analyse_all(contexts), strict=True)
        )

    @override
    def analyse_all(
        self,
        contexts: Iterable[Context],
    ) -> Iterator[tuple[Token, ...]]:
        """
        Hand back the reading made of each context, reading what is new.

        Args:
            contexts: The texts to read, or the words they were split into.

        Yields:
            The words of one context, in the order the contexts came in.
        """
        for context in contexts:
            if context not in self._readings:
                self.remember([context])

            yield self._readings[context]


@dataclass(frozen=True, slots=True)
class Score:
    """
    What one engine made of one set of queries.

    Attributes:
        returned: How many occurrences it returned.
        recorded: How many the treebank records.
        agreed: How many of them are the same occurrence.
        seconds: How long it took to read the sentences.
        words: How many words it read.
    """

    returned: int
    recorded: int
    agreed: int
    seconds: float
    words: int

    @property
    def precision(
        self,
    ) -> float:
        """How much of what it returned the treebank records."""
        return self.agreed / self.returned if self.returned else 0.0

    @property
    def recall(
        self,
    ) -> float:
        """How much of what the treebank records it returned."""
        return self.agreed / self.recorded if self.recorded else 0.0

    @property
    def f1(
        self,
    ) -> float:
        """The two at once, neither weighing more than the other."""
        total = self.precision + self.recall

        return 2 * self.precision * self.recall / total if total else 0.0

    @property
    def rate(
        self,
    ) -> float:
        """How many words a second it read, loading not counted."""
        return self.words / self.seconds if self.seconds else 0.0


@dataclass(frozen=True, slots=True)
class Table:
    """
    One table of the evaluation.

    Attributes:
        caption: What the table is about.
        note: What was asked to make it.
        columns: The heading of each column, the first being the row's name.
        rows: The rows, already written out.
    """

    caption: str
    note: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


def sample_queries(
    sentences: Iterable[Sentence],
    count: int,
    seed: int,
) -> tuple[Query, ...]:
    """
    Draw the lemmas to ask for out of the treebank itself.

    A lemma is drawn as often as it is written, so the sample holds the
    lemmas a caller is likely to look for.

    Args:
        sentences: The sentences to draw from.
        count: How many draws to make.
        seed: What to draw them with.

    Returns:
        The queries, each a lemma under one tag and each asked once.
    """
    drawn = [
        (word.lemma, POS(word.pos))
        for sentence in sentences
        for word in sentence.words
        if word.pos in QUERIED
    ]

    queries = {
        Query(lemma=lemma, pos=pos)
        for lemma, pos in random.Random(seed).sample(drawn, count)
    }

    return tuple(queries)


def with_forms(
    queries: Iterable[Query],
) -> tuple[Query, ...]:
    """
    Give every query the forms LemmInflect writes its lemma as.

    Its lexicon holds nothing for a rare word, and the rare words are the
    ones an engine reads wrongly, so what it holds none of it generates.

    Args:
        queries: The lemmas asked for.

    Returns:
        The same queries, each carrying its inflections.
    """
    return tuple(
        Query(
            lemma=query.lemma,
            pos=query.pos,
            forms=frozenset(
                form
                for written in _inflections(query.lemma, query.pos).values()
                for form in written
            ),
        )
        for query in queries
    )


def _inflections(
    lemma: str,
    pos: POS | None,
) -> dict[str, tuple[str, ...]]:
    """
    Read every form of one lemma, from the lexicon or from the rules.

    Args:
        lemma: The lemma to inflect.
        pos: The tag to inflect it under.

    Returns:
        The forms, under the tag each carries.
    """
    held = cast("dict[str, tuple[str, ...]]", getAllInflections(lemma, pos))

    return held or cast("dict[str, tuple[str, ...]]", getAllInflectionsOOV(lemma, pos))


def recorded_words(
    sentences: Sequence[Sentence],
    queries: Iterable[Query],
) -> Iterator[tuple[int, Word]]:
    """
    Read out of the treebank every word the queries ask for.

    Args:
        sentences: The sentences the study is run over.
        queries: The lemmas asked for.

    Yields:
        One word at a time, with the sentence it falls in.
    """
    wanted = {
        (normalise(query.lemma), tag)
        for query in queries
        for tag in (POS if query.pos is None else (query.pos,))
    }

    for index, sentence in enumerate(sentences):
        for word in sentence.words:
            if (normalise(word.lemma), POS(word.pos)) in wanted:
                yield index, word


def read_sentences(
    engine: Remembered,
    sentences: Iterable[Sentence],
) -> float:
    """
    Read every sentence once, so that a query costs only the looking.

    Args:
        engine: The engine to read with, which remembers what it reads.
        sentences: The sentences to read.

    Returns:
        How long the reading took.
    """
    started = perf_counter()
    engine.remember([sentence.text for sentence in sentences])

    return perf_counter() - started


def score(
    engine: Engine,
    sentences: Sequence[Sentence],
    queries: Sequence[Query],
    recorded: set[Occurrence],
    seconds: float,
) -> Score:
    """
    Put one set of queries to a reading and count where it agrees.

    Args:
        engine: The engine holding the reading.
        sentences: The sentences it read.
        queries: The lemmas to ask for.
        recorded: The occurrences the treebank records.
        seconds: How long the reading took.

    Returns:
        What it returned, against what there was to return.
    """
    locator = Locator(engine)

    returned = {
        (index, *match.offsets)
        for index, sentence in enumerate(sentences)
        for match in locator.find(sentence.text, queries)
        if match.offsets is not None
    }

    return Score(
        returned=len(returned),
        recorded=len(recorded),
        agreed=len(returned & recorded),
        seconds=seconds,
        words=sum(len(sentence.words) for sentence in sentences),
    )


def evaluate(
    names: Iterable[str],
    sentences: Sequence[Sentence],
    questions: Mapping[str, Sequence[Query]],
) -> dict[str, dict[str, Score]]:
    """
    Run every engine over the sentences, under each set of queries.

    Args:
        names: The engines to run, as ENGINES names them.
        sentences: The sentences to read.
        questions: The lemmas to ask for, under what the asking is called.

    Returns:
        What each engine scored, by set of queries and then by engine.
    """
    recorded = {
        caption: {
            (index, *word.offsets) for index, word in recorded_words(sentences, queries)
        }
        for caption, queries in questions.items()
    }

    scored: dict[str, dict[str, Score]] = {caption: {} for caption in questions}

    for name in names:
        say(f"{name}: loading, then reading {len(sentences):,} sentences")

        engine = Remembered(ENGINES[name]())
        seconds = read_sentences(engine, sentences)

        for caption, queries in questions.items():
            scored[caption][name] = score(
                engine, sentences, queries, recorded[caption], seconds
            )

        figures = scored[next(iter(questions))][name]
        say(f"{name}: {figures.seconds:,.0f}s at {figures.rate:,.0f} words/s")

    return scored


def tabulate_reading(
    scores: Mapping[str, Score],
) -> Table:
    """
    Write out what reading the sentences cost each engine.

    Reading is done once and asked twice, so it is reported apart from what
    the asking found.

    Args:
        scores: What each engine scored, under its name.

    Returns:
        The table.
    """
    words = next(iter(scores.values())).words

    return Table(
        caption="Reading",
        note=f"{words:,} words, the loading of a model not counted.",
        columns=("Engine", "Seconds", "Words/s"),
        rows=tuple(
            (name, f"{figures.seconds:,.1f}", f"{figures.rate:,.0f}")
            for name, figures in scores.items()
        ),
    )


def tabulate(
    caption: str,
    scores: Mapping[str, Score],
    queries: Sequence[Query],
) -> Table:
    """
    Write out what the engines scored under one set of queries.

    Args:
        caption: What the queries have in common.
        scores: What each engine scored, under its name.
        queries: The lemmas that were asked for.

    Returns:
        The table.
    """
    recorded = next(iter(scores.values())).recorded

    return Table(
        caption=caption,
        note=f"{len(queries):,} lemmas asked for, {recorded:,} occurrences.",
        columns=("Engine", "Precision", "Recall", "F1"),
        rows=tuple(
            (
                name,
                f"{figures.precision:.4f}",
                f"{figures.recall:.4f}",
                f"{figures.f1:.4f}",
            )
            for name, figures in scores.items()
        ),
    )


def to_markdown(
    tables: Iterable[Table],
    note: str,
) -> str:
    """
    Write the evaluation as Markdown, for a document to take as it stands.

    Args:
        tables: The tables to write.
        note: What to say before them.

    Returns:
        The document.
    """
    lines = ["# Evaluation", "", note, ""]

    for table in tables:
        lines += [
            f"## {table.caption}",
            "",
            table.note,
            "",
            f"| {' | '.join(table.columns)} |",
            f"|{'|'.join(' --- ' for _ in table.columns)}|",
        ]
        lines += [f"| {' | '.join(row)} |" for row in table.rows]
        lines.append("")

    return "\n".join(lines)


def write(
    tables: Iterable[Table],
    note: str,
    directory: Path,
) -> Path:
    """
    Write the evaluation to a directory, under the moment it was written.

    Args:
        tables: The tables to write.
        note: What to say before them.
        directory: Where the document goes.

    Returns:
        The document written.
    """
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / f"{datetime.now(UTC):{STAMP}}.md"
    _ = path.write_text(to_markdown(tables, note), encoding="utf-8")

    return path


class Arguments(argparse.Namespace):
    """
    What the command line settles.

    Attributes:
        treebank: The section to run against.
        engine: The engines to run, or None for all of them.
        queries: How many draws to make.
        seed: What to draw them with.
        sentences: How many sentences to read, or None for the section.
        forms: Whether to give every query the forms of its lemma.
        to: Which directory the evaluation is written to.
    """

    treebank: str = "ewt"
    engine: list[str] | None = None
    queries: int = QUERIES
    seed: int = SEED
    sentences: int | None = None
    forms: bool = False
    to: Path = EVALUATIONS


def read_arguments() -> Arguments:
    """
    Read what the command line settles, falling back on the defaults above.

    Returns:
        The treebank to read, the engines to run and the sample to draw.
    """
    parser = argparse.ArgumentParser(description="Measure the engines.")

    _ = parser.add_argument(
        "--treebank",
        choices=sorted(treebanks.TREEBANKS),
        help=f"section to run against (default: {Arguments.treebank})",
    )
    _ = parser.add_argument(
        "--engine",
        action="append",
        choices=sorted(ENGINES),
        help="engine to run; repeat to name several (default: all of them)",
    )
    _ = parser.add_argument(
        "--queries",
        type=int,
        help=f"how many lemmas to draw (default: {QUERIES})",
    )
    _ = parser.add_argument(
        "--seed",
        type=int,
        help=f"what to draw them with (default: {SEED})",
    )
    _ = parser.add_argument(
        "--sentences",
        type=int,
        help="read this many sentences only, for a quick pass",
    )
    _ = parser.add_argument(
        "--forms",
        action="store_true",
        help="give every query the forms of its lemma, as a fallback",
    )
    _ = parser.add_argument(
        "--to",
        type=Path,
        help="write the evaluation to this directory",
    )

    return parser.parse_args(namespace=Arguments())


def main() -> None:
    """
    Run the engines over a treebank and write what they scored.
    """
    arguments = read_arguments()
    started = perf_counter()

    sentences = list(treebanks.read(treebanks.fetch(arguments.treebank)))[
        : arguments.sentences
    ]

    queries = sample_queries(sentences, arguments.queries, arguments.seed)

    questions = {
        OWN_TAG: queries,
        ANY_TAG: tuple({Query(lemma=query.lemma) for query in queries}),
    }

    if arguments.forms:
        questions = {caption: with_forms(asked) for caption, asked in questions.items()}

    scored = evaluate(arguments.engine or sorted(ENGINES), sentences, questions)

    tables = [
        tabulate_reading(scored[OWN_TAG]),
        *(
            tabulate(caption, scored[caption], questions[caption])
            for caption in questions
        ),
    ]

    note = (
        f"{arguments.treebank}, {len(sentences):,} sentences, "
        f"{arguments.queries:,} draws seeded with {arguments.seed}."
    )

    written = write(tables, note, arguments.to)

    say(f"Wrote {written} in {perf_counter() - started:,.0f}s")


if __name__ == "__main__":
    main()
