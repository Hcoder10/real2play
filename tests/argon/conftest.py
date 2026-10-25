"""Argon tests are hermetic: no real Gemini, no hardware, no GPU."""
import pytest


@pytest.fixture(autouse=True)
def offline_gemini(monkeypatch):
    from r2p import gemini

    monkeypatch.setenv("R2P_MODEL_MAIN", "gemini-4-argon")
    monkeypatch.setenv("R2P_MODEL_SMALL", "gemini-3.8-flash")
    for name in ("R2P_CACHE_ONLY", "R2P_ALLOW_FALLBACK", "R2P_ALLOW_FLASH", "R2P_THINKING"):
        monkeypatch.delenv(name, raising=False)

    def request(self, model, parts, config):
        if self.transport is None:
            raise AssertionError("Real Gemini requests are forbidden in tests")
        return self.transport(model=model, parts=parts, config=config)
    monkeypatch.setattr(gemini.Gemini, "_request", request)
    def no_live_client(self):
        raise AssertionError("Real Interactions clients are forbidden in tests")
    # _request is separately restored only in SDK-boundary tests, which inject a client.
    # Keep _client available there through an explicit per-instance monkeypatch.
    from r2p.agents import ManagedAgent
    def managed_client(self):
        if self.transport is None:
            return no_live_client(self)
        from contextlib import nullcontext
        from types import SimpleNamespace
        return nullcontext(SimpleNamespace(interactions=self.transport))
    monkeypatch.setattr(ManagedAgent, "_client", managed_client)
