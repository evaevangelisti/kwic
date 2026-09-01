# Key Word In Context

Locates the lemmas of a query in the context that attests them.

<!-- installation -->

## Installation

### Requirements

- Python 3.12 or later

### From PyPI

```sh
pip install kwic
```

A search reads with spaCy, whose pipelines are published apart from the library:

```sh
python -m spacy download en_core_web_sm
```

The other engines are extras:

```sh
pip install "kwic[stanza]"
pip install "kwic[lemminflect]"
```

<!-- usage -->

## Usage

A search takes one context and the lemmas to look for.

```python
from kwic import Locator, POS, Query

locator = Locator()

locator.find("She found the keys she had lost.", [Query("find", POS.VERB)])
# (Match(lemma='find', pos=POS.VERB, form='found', word_index=1, offsets=(4, 9)),)
```

### Query

| Field | Default | |
| --- | --- | --- |
| `lemma` | | Dictionary form to look for, in whatever case |
| `pos` | `None` | Tag a one-word occurrence must carry; a longer lemma is not narrowed |
| `forms` | `()` | How else the lemma is written, taken where the engine read another |

The engine cuts the lemma into words, and a space and a hyphen are one to it.

```python
locator.find("It was hunky-dory.", [Query("hunky dory")])
# (Match(lemma='hunky dory', pos=POS.ADJ, form='hunky-dory', word_index=2, offsets=(7, 17)),)
```

A phrasal verb written apart runs from the verb to the particle.

```python
locator.find("She gave the money up.", [Query("give up")])
# (Match(lemma='give up', pos=POS.VERB, form='gave the money up', word_index=1, offsets=(4, 21)),)
```

### Match

| Field | |
| --- | --- |
| `lemma` | The lemma you asked for, as you wrote it |
| `pos` | The tag it carries |
| `form` | How it is written |
| `word_index` | Where it opens among the words, from zero |
| `offsets` | Where it falls in the text, half-open and in code points |

### Contexts

A context is a text, or the words it was split into. The second has no range.

```python
locator.find(("She", "found", "the", "keys"), [Query("find")])
# (Match(lemma='find', pos=POS.VERB, form='found', word_index=1, offsets=None),)
```

### Many contexts

`find_all` takes context and lemmas in pairs, and reads a batch at a time.

```python
for occurrences in locator.find_all(searches):
    ...
```

### Engines

An engine reads the context; the search reads the engine.

```python
from kwic.engines.stanza import StanzaEngine

locator = Locator(StanzaEngine())
```

| Engine | Reads with | Install |
| --- | --- | --- |
| `SpacyEngine` | a spaCy pipeline, its English lemmatiser rules over the tag | `spacy` |
| `StanzaEngine` | Stanza, a dictionary with a neural model behind it | `kwic[stanza]` |
| `LemmInflectEngine` | spaCy for the tags, LemmInflect for the lemmas | `kwic[lemminflect]` |

An extra is imported from the module wrapping it, so a package without it still loads.

`SpacyEngine` takes the pipeline to load, `StanzaEngine` the language. Both parse unless told otherwise:

| Parser | Buys | Costs |
| --- | --- | --- |
| spaCy | phrasal verbs apart, and several universal tags | a tenth of a reading |
| Stanza | phrasal verbs apart | half its speed |

```python
Locator(SpacyEngine(parse=False))
```

<!-- accuracy -->

## Accuracy

English-EWT test section: 2,077 sentences, 417 lemmas, 3,619 occurrences.

| Engine | Precision | Recall | F1 |
| --- | --- | --- | --- |
| `StanzaEngine` | 0.975 | 0.963 | 0.969 |
| `SpacyEngine`, `en_core_web_trf` | 0.981 | 0.905 | 0.942 |
| `SpacyEngine`, `en_core_web_lg` | 0.978 | 0.891 | 0.933 |
| `SpacyEngine`, `en_core_web_sm` | 0.978 | 0.882 | 0.928 |
| `LemmInflectEngine` | 0.976 | 0.872 | 0.921 |
