"""
Fixtures the whole suite shares.

A test states a property: whatever a generator draws, this holds of it. The
generators live in strategies.py and the engine standing in for a model in
stubs.py, so that a test file holds properties and little else. An example is
spelled out where one reading is the point of the test.

A feature is exercised from outside, through what the package exports. An
engine is part of that: taking one of its own is what a search is for, and
the one the suite hands it reads nothing at all, so that a property about a
search is not a property about a model.

Where a model is the point, it is loaded once for the whole run and the test
is marked pipeline, so that a run can leave it out.

The test tree mirrors the package: tests/engines/test_spacy.py covers
src/kwic/engines/spacy.py, and a module at the top of the tree covers one of
the same name in src/kwic. What every directory reads lives here; what one
directory alone needs belongs in a conftest.py of its own.

A test is documented in a single line, its parameters being fixtures rather
than arguments. A fixture is documented like any other function.
"""

import pytest
from hypothesis import settings

from kwic import Locator, SpacyEngine

# Loading a pipeline is timed by the machine it runs on, and how long one
# example took there says nothing about the code.
settings.register_profile("kwic", deadline=None)
settings.load_profile("kwic")


@pytest.fixture(scope="session")
def engine() -> SpacyEngine:
    """
    Load the small English pipeline, once for the whole run.

    Returns:
        An engine reading with it.
    """
    return SpacyEngine()


@pytest.fixture(scope="session")
def locator(
    engine: SpacyEngine,
) -> Locator:
    """
    Hand the pipeline to a search.

    Args:
        engine: The engine to read with.

    Returns:
        A search reading with it.
    """
    return Locator(engine)
