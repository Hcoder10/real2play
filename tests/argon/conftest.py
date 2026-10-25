"""Argon tests are hermetic: no real Gemini, no hardware, no GPU."""
import pytest


@pytest.fixture(autouse=True)
def offline_gemini(monkeypatch):
    from r2p import gemini

    monkeypatch.setenv("R2P_MODEL_MAIN", "gemini-4-argon")
    monkeypatch.setenv("R2P_MODEL_SMALL", "gemini-3.8-flash")
    for name in ("R2P_CACHE_ONLY", "R2P_ALLOW_FALLBACK", "R2P_THINKING"):
        monkeypatch.delenv(name, raising=False)

    def request(self, model, parts, config):
        if self.transport is None:
            raise AssertionError("Real Gemini requests are forbidden in tests")
        return self.transport(model=model, parts=parts, config=config)
    monkeypatch.setattr(gemini.Gemini, "_request", request)
