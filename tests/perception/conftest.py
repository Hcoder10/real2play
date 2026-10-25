import os

import pytest


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("R2P_CACHE_ONLY", "1")
    # Every detector test injects an oracle/fake. Even accidental Gemini imports
    # cannot make a paid request under the cache-only guard.
