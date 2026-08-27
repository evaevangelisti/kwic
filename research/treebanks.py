"""
Reading the treebanks the studies are measured against.

A treebank is the only gold there is here: it says, for every word of every
sentence, what lemma and what tag a linguist read it under. A study puts the
same question to an engine and counts where the two disagree.

Only the test sections are fetched. They are what a model was held back
from, or ought to have been, and the sections it was trained on would say
more about the treebank than about the engine.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from http.client import HTTPResponse
from pathlib import Path
from typing import cast
from urllib.request import urlopen

DATA = Path(__file__).resolve().parent.parent / "data"
"""Where the treebanks the studies draw from are kept, outside version
control."""

TREEBANKS = {
    "ewt": "UD_English-EWT/master/en_ewt-ud-test.conllu",
    "gum": "UD_English-GUM/master/en_gum-ud-test.conllu",
}
"""The test sections a study may be run against, by the name it calls them."""

TREEBANK_URL = "https://raw.githubusercontent.com/UniversalDependencies/{path}"
"""Where a treebank is published, one file per section."""

TEXT = "# text = "
"""What the line carrying a sentence opens with."""

MISSING = "_"
"""What a column holds where the treebank has nothing to put in it."""

type _Row = tuple[str, str, str, str]
"""One line of a section, cut down to the columns a study draws on."""


@dataclass(frozen=True, slots=True)
class Word:
    """
    One word of a sentence, as the treebank reads it.

    Attributes:
        form: The word as it is written.
        lemma: The lemma the treebank reads it under.
        pos: Its universal tag.
        offsets: Where it falls in the sentence. A word spelled inside a
        larger token, as n't is inside don't, is given the range of the
        token, that being the only range the sentence has for it.
    """

    form: str
    lemma: str
    pos: str
    offsets: tuple[int, int]


@dataclass(frozen=True, slots=True)
class Sentence:
    """
    One sentence and the reading of every word in it.

    Attributes:
        text: The sentence as it is written.
        words: Its words, in order.
    """

    text: str
    words: tuple[Word, ...]


def fetch(
    name: str,
) -> Path:
    """
    Fetch one treebank, keeping it for the runs to come.

    Args:
        name: What the study calls the treebank, such as ewt.

    Returns:
        Where the section is kept.

    Raises:
        SystemExit: If no treebank is published under that name.
    """
    if name not in TREEBANKS:
        raise SystemExit(f"No treebank called {name}; try {', '.join(TREEBANKS)}")

    path = DATA / f"{name}.conllu"

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

        # What urlopen hands back depends on the scheme it was given, and is
        # typed as anything at all. Only one scheme is ever asked for here.
        opened = cast(
            "HTTPResponse", urlopen(TREEBANK_URL.format(path=TREEBANKS[name]))
        )

        with opened as response:
            _ = path.write_bytes(response.read())

    return path


def read(
    path: Path,
) -> Iterator[Sentence]:
    """
    Read a section into the sentences it holds.

    A sentence whose words cannot all be found in it is left out. The
    treebanks normalise a handful of them, and a word that is not written
    where it is said to be is one no study can ask an engine about.

    Args:
        path: The section to read.

    Yields:
        One sentence at a time, in the order they are written.
    """
    for block in path.read_text(encoding="utf-8").split("\n\n"):
        sentence = _sentence(block)

        if sentence is not None:
            yield sentence


def _sentence(
    block: str,
) -> Sentence | None:
    """
    Read one block of the section into a sentence.

    The tokens are located first and the words read off them after, a word
    being written inside a token rather than beside it.

    Args:
        block: The lines between two blank ones.

    Returns:
        The sentence, or None where it holds no words or where one of them
        is not written in it.
    """
    text = ""
    rows: list[_Row] = []

    for line in block.splitlines():
        if line.startswith(TEXT):
            text = line.removeprefix(TEXT)
        elif line and not line.startswith("#"):
            rows.append(_columns(line))

    offsets = _locate_tokens(text, rows)

    if offsets is None:
        return None

    words = tuple(
        Word(form=form, lemma=lemma, pos=pos, offsets=offsets[index])
        for index, form, lemma, pos in rows
        if index in offsets
    )

    return Sentence(text=text, words=words) if words else None


def _locate_tokens(
    text: str,
    rows: list[_Row],
) -> dict[str, tuple[int, int]] | None:
    """
    Find where every token of a sentence is written.

    Args:
        text: The sentence.
        rows: Its lines, in the order the treebank writes them.

    Returns:
        The range each word occupies, under the index the treebank gives it,
        several words sharing the range of the token they are written in. Or
        None where a token is not written in the sentence at all.
    """
    covered = {
        number for index, *_ in rows if "-" in index for number in _covered(index)
    }

    offsets: dict[str, tuple[int, int]] = {}
    position = 0

    for index, form, _, _ in rows:
        # An empty node is what the enhanced graph needs and the sentence
        # does not write; a word inside a token is written in the token.
        if "." in index or index in covered:
            continue

        located = _locate(text, form, position)

        if located is None:
            return None

        position = located[1]

        for number in _covered(index):
            offsets[number] = located

    return offsets


def _covered(
    index: str,
) -> tuple[str, ...]:
    """
    Read the words one index stands for.

    Args:
        index: The index of a word, or the range of a token holding several.

    Returns:
        The indices of the words it covers, which is itself where it is not
        a range.
    """
    if "-" not in index:
        return (index,)

    first, last = (int(part) for part in index.split("-"))

    return tuple(str(number) for number in range(first, last + 1))


def _columns(
    line: str,
) -> _Row:
    """
    Read the columns a study draws on out of one line.

    Args:
        line: One word of a sentence, tab separated.

    Returns:
        Its index, its form, its lemma and its universal tag. A lemma the
        treebank leaves out is taken to be the form.
    """
    index, form, lemma, pos = line.split("\t")[:4]

    return index, form, form if lemma == MISSING else lemma, pos


def _locate(
    text: str,
    form: str,
    position: int,
) -> tuple[int, int] | None:
    """
    Find where one form is written, reading on from the last one.

    Args:
        text: The sentence.
        form: The form to find.
        position: Where the form before it closed.

    Returns:
        The range it occupies, or None where it is not written there.
    """
    start = text.find(form, position)

    if start < 0:
        return None

    return start, start + len(form)
