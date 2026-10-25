"""Antigravity lifecycle and reviewed file bundles. No publishing or credential forwarding."""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import PurePosixPath
import re
from types import SimpleNamespace

from r2p import gemini

AGENT = "antigravity-preview-09-2026"
MODEL = "gemini-4-argon"


def assert_no_secrets(text):
    """Check before submission, including tokens in caller-provided source/config."""
    for name, value in os.environ.items():
        if re.search(r"(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", name, re.I) and len(value) >= 8 and value in text:
            raise ValueError(f"Sandbox input contains a secret from {name}")
    if re.search(r"AIza[\w-]{30,}|(?:gh[pousr]_|github_pat_|sk-[A-Za-z0-9])[\w-]{15,}|-----BEGIN .*PRIVATE KEY-----", text):
        raise ValueError("Sandbox input contains a credential-like value")


def safe_path(name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or "\x00" in name:
        raise ValueError("Invalid generated path")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError("Generated paths must be relative without traversal")
    if any(part.startswith(".") or part.rstrip(" .") != part for part in path.parts):
        raise ValueError("Hidden/ambiguous generated paths are forbidden")
    if any(part.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)],
                                          *[f"LPT{i}" for i in range(1, 10)]} for part in path.parts):
        raise ValueError("Reserved generated path")
    return path


def validate_bundle(value):
    if not isinstance(value, dict) or not isinstance(value.get("files"), dict) or not value["files"]:
        raise gemini.JSONError("Agent must return nonempty files, tests and report")
    if not isinstance(value.get("tests"), list) or not value["tests"] or not isinstance(value.get("report"), str):
        raise gemini.JSONError("Agent must return tests and a report")
    total = 0
    seen = set()
    for name, content in value["files"].items():
        safe_path(name)
        if name.casefold() in seen:
            raise gemini.JSONError("Duplicate generated path")
        seen.add(name.casefold())
        if not isinstance(content, str):
            raise gemini.JSONError("File contents must be UTF-8 strings")
        assert_no_secrets(content)
        total += len(content.encode("utf-8"))
    if total > 10_000_000 or len(seen) > 100:
        raise gemini.JSONError("Agent file bundle exceeds limits")
    for test in value["tests"]:
        if not isinstance(test, dict) or not isinstance(test.get("name"), str) or type(test.get("passed")) is not bool:
            raise gemini.JSONError("tests require name and boolean passed")
    return value


class ManagedAgent(gemini.Gemini):
    """Start once; retain interaction IDs so timed-out runs can be resumed.

    `transport` is an injectable object exposing create/get, not an LLM responder.
    Every create/get consumes max_requests; cache-only never constructs a client.
    """

    def __init__(self, *args, source_files=None, **kwargs):
        self.source_files = dict(source_files or {})
        for name, content in self.source_files.items():
            safe_path(name)
            if not isinstance(content, str):
                raise ValueError("Sandbox sources must be UTF-8 text")
            assert_no_secrets(content)
        super().__init__(*args, **kwargs)

    def _client(self):
        if self.transport is not None:
            return nullcontext(SimpleNamespace(interactions=self.transport))
        return super()._client()

    def _body(self, prompt):
        assert_no_secrets(prompt)
        environment = "remote"
        if self.source_files:
            for name, content in self.source_files.items():
                safe_path(name)
                assert_no_secrets(content)
            # SDK 2.33 Source/Environment: avoid asking a model to retype fixtures.
            environment = {"type": "remote", "sources": [
                {"type": "inline", "target": "/workspace/" + name, "content": content}
                for name, content in sorted(self.source_files.items())]}
        return {"agent": AGENT, "agent_config": {"type": "antigravity", "model": MODEL},
                "input": prompt, "environment": environment, "background": True}

    def _path(self, prompt):
        key = hashlib.sha256(gemini._canonical(self._body(prompt)).encode()).hexdigest()
        return self.cache_dir / (key + ".agent.json")

    def start(self, prompt, *, tag="managed_agent"):
        body = self._body(prompt)
        path = self._path(prompt)
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("status") == "completed":
                self._log({"tag": tag, "api": "interactions", "agent": AGENT, "served_by": MODEL,
                           "cached": True, "latency": 0, "interaction_id": saved.get("id"), "usage": saved.get("usage")})
                return saved
            if os.getenv("R2P_CACHE_ONLY") != "1":
                return saved
        if os.getenv("R2P_CACHE_ONLY") == "1":
            raise gemini.CacheMiss("Cache-only managed-agent miss")
        start = self.clock()
        record = {"api": "interactions", "operation": "create", "agent": AGENT, "tag": tag,
                  "served_by": None, "cached": False, "usage": None}
        try:
            self._reserve()
            with self._client() as client:
                reply = client.interactions.create(**body, timeout=self.timeout_s)
            if not gemini.field(reply, "id"):
                raise gemini.GeminiError("Agent response has no interaction id")
            record.update(interaction_id=gemini.field(reply, "id"), status=gemini.field(reply, "status"),
                          served_by=gemini.field(reply, "model") or MODEL, usage=gemini.plain(gemini.field(reply, "usage")))
            gemini.atomic_json(path, gemini.plain(reply))
            return reply
        except Exception as exc:
            record["error"] = type(exc).__name__
            if gemini._status(exc) == 429 and gemini._daily_quota(exc):
                raise gemini.QuotaError("Antigravity daily quota exhausted") from None
            if isinstance(exc, gemini.GeminiError):
                raise
            # No blind retries: a lost create response may already have started a sandbox.
            raise gemini.GeminiError(f"Agent start failed: {gemini._safe(exc)}") from None
        finally:
            record["latency"] = self.clock() - start
            self._log(record)

    def poll(self, interaction, *, tag="managed_agent", timeout_s=None):
        if isinstance(interaction, str):
            interaction = {"id": interaction, "status": "in_progress"}
        if gemini.field(interaction, "status") == "completed":
            return interaction
        if os.getenv("R2P_CACHE_ONLY") == "1":
            raise gemini.CacheMiss("Cache-only forbids managed-agent polling")
        with self._client() as client:
            return self._poll(client, interaction, tag=tag, deadline=self.clock() + (timeout_s or self.timeout_s))

    def collect(self, interaction):
        if gemini.field(interaction, "status") != "completed":
            raise gemini.GeminiError("Only completed interactions can be collected")
        text = gemini.field(interaction, "output_text")
        bundle = validate_bundle(gemini.parse_json(text))
        artifacts = []
        for step in gemini.field(interaction, "steps", []) or []:
            if gemini.field(step, "type") == "model_output":
                for content in gemini.field(step, "content", []) or []:
                    if gemini.field(content, "type") != "text":
                        artifacts.append(gemini.plain(content))
        return {**bundle, "output_text": text, "artifacts": artifacts}

    def run(self, prompt, *, tag="managed_agent"):
        start = self.clock()
        requests_before = self.requests_used
        interaction = self.start(prompt, tag=tag)
        interaction = self.poll(interaction, tag=tag)
        gemini.atomic_json(self._path(prompt), gemini.plain(interaction))
        bundle = self.collect(interaction)
        record = {"api": "interactions", "operation": "collect", "tag": tag, "agent": AGENT,
                  "served_by": gemini.field(interaction, "model") or MODEL, "cached": self.requests_used == requests_before,
                  "latency": self.clock() - start, "usage": gemini.plain(gemini.field(interaction, "usage")),
                  "interaction_id": gemini.field(interaction, "id"), "request_key": self._path(prompt).stem,
                  "files": {name: hashlib.sha256(content.encode("utf-8")).hexdigest()
                            for name, content in bundle["files"].items()}}
        self._log(record)
        return {**bundle, "record": record}


class FakeAgent(ManagedAgent):
    """Exercise the full lifecycle, serialization and gates without a Google client."""
    def __init__(self, bundle, **kwargs):
        self.bundle = bundle
        self.requests = []
        super().__init__(transport=self, sleep=lambda _: None, **kwargs)

    def create(self, **kwargs):
        self.requests.append(("create", kwargs))
        return {"id": "fake-antigravity", "status": "in_progress", "model": MODEL}

    def get(self, **kwargs):
        self.requests.append(("get", kwargs))
        return {"id": "fake-antigravity", "status": "completed", "model": MODEL,
                "output_text": json.dumps(self.bundle), "usage": {"total_tokens": 0}}
