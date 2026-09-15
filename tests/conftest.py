"""
Fixtures and Hypothesis settings shared by the test suite.
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
