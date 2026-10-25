"""Role-aware Gemini calls with an offline, content-addressed cache.

Only this module talks to google-genai. Images are BGR (OpenCV convention).
The transport is injectable; FakeGemini never creates a Google client.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any


class GeminiError(RuntimeError):
    pass


class QuotaError(GeminiError):
    """A daily or zero-limit quota; retrying this run cannot help."""


class CacheMiss(GeminiError):
    pass


class JSONError(GeminiError):
    pass


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix="." + path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False, default=_json_default)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _json_default(value):
    if isinstance(value, bytes):
        return {"base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def _load_dotenv():
    """GEMINI_API_KEY (and other unset vars) from the gitignored repo .env; real environment wins."""
    env = Path(__file__).resolve().parents[1] / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            k, sep, v = line.strip().partition("=")
            if sep and k and not k.startswith("#") and k not in os.environ:
                os.environ[k] = v.strip().strip('"').strip("'")


_load_dotenv()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_json_default)


def _safe(text):
    key = os.environ.get("GEMINI_API_KEY")
    return str(text).replace(key, "[REDACTED]") if key else str(text)


def model_for(role):
    if role not in ("main", "small"):
        raise ValueError("role must be main or small")
    return os.getenv("R2P_MODEL_" + role.upper(), "gemini-4-argon" if role == "main" else "gemini-3.8-flash")


def image_part(bgr):
    import cv2
    import numpy as np

    pixels = np.asarray(bgr)
    if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint8:
        raise ValueError("image_part requires an H x W x 3 uint8 BGR image")
    ok, data = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("Image could not be encoded")
    return {"inline_data": {"mime_type": "image/jpeg", "data": data.tobytes()}}


def frames_to_video(frames, path, fps=30):
    """Encode an entire ordered BGR frame sequence, without uploading anything."""
    import cv2

    if fps <= 0:
        raise ValueError("fps must be positive")
    iterator = iter(frames)
    first = next(iterator, None)
    if first is None:
        raise ValueError("Cannot encode an empty episode")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("MP4 encoder unavailable")
    try:
        writer.write(first)
        for frame in iterator:
            if frame.shape != first.shape:
                raise ValueError("All video frames must have the same dimensions")
            writer.write(frame)
    finally:
        writer.release()
    return path


def video_part(path):
    path = Path(path)
    if path.suffix.lower() not in (".mp4", ".mov", ".webm"):
        raise ValueError("Expected an MP4, MOV or WebM video")
    mime = {".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm"}[path.suffix.lower()]
    return {"inline_data": {"mime_type": mime, "data": path.read_bytes()}}


def parse_json(text):
    stripped = str(text).strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)[:-3].strip()
    try:
        return json.loads(stripped, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (ValueError, TypeError) as exc:
        raise JSONError("Invalid JSON: " + str(exc)) from None


def schema_errors(value, schema, path="$"):
    """Validate the JSON-Schema subset used by our request contracts, offline."""
    if not schema:
        return []
    if hasattr(schema, "model_validate"):
        try:
            schema.model_validate(value)
            return []
        except ValueError as exc:
            return [str(exc)]
    errors = []
    for keyword in ("anyOf", "oneOf"):
        if keyword in schema:
            count = sum(not schema_errors(value, sub, path) for sub in schema[keyword])
            if count == 0 or (keyword == "oneOf" and count != 1):
                return [f"{path}: does not match {keyword}"]
    kind = schema.get("type")
    kinds = kind if isinstance(kind, list) else [kind] if kind else []
    checks = {"object": lambda x: isinstance(x, dict), "array": lambda x: isinstance(x, list),
              "string": lambda x: isinstance(x, str), "boolean": lambda x: isinstance(x, bool),
              "integer": lambda x: isinstance(x, int) and not isinstance(x, bool),
              "number": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
              "null": lambda x: x is None}
    if kinds and not any(checks[str(k).lower()](value) for k in kinds):
        return [f"{path}: expected {kind}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: expected one of {schema['enum']}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}.{key}: required")
        for key, item in value.items():
            if key in schema.get("properties", {}):
                errors.extend(schema_errors(item, schema["properties"][key], path + "." + key))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}.{key}: not permitted")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", float("inf")):
            errors.append(f"{path}: wrong array length")
        for i, item in enumerate(value):
            errors.extend(schema_errors(item, schema.get("items", {}), f"{path}[{i}]"))
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        if value < schema.get("minimum", -float("inf")) or value > schema.get("maximum", float("inf")):
            errors.append(f"{path}: outside allowed range")
    return errors


def _daily_quota(exc):
    # "billing" is present in ordinary RPM errors too. Never classify on it.
    message = str(exc)
    return bool(re.search(r"per[\s_-]*day|requestsperday|\blimit\s*[:=\-]?\s*0(?:\D|$)|"
                          r'"quotaValue"\s*:\s*"?0"?', message, re.I))


def _status(exc):
    for name in ("code", "status_code"):
        value = getattr(exc, name, None)
        if value == 429 or str(value) == "429":
            return 429
    return 429 if re.search(r"\b429\b|RESOURCE_EXHAUSTED", str(exc)) else None


class Gemini:
    def __init__(self, cache_dir=None, *, transport=None, sleep=time.sleep, retries=3):
        self.cache_dir = Path(cache_dir or os.getenv("R2P_GEMINI_CACHE", "cache/gemini"))
        self.transport = transport
        self.sleep = sleep
        self.retries = retries
        self.records = []

    def _request(self, model, parts, config):
        if self.transport is not None:
            return self.transport(model=model, parts=parts, config=config)
        from google import genai
        from google.genai import types

        if not os.getenv("GEMINI_API_KEY"):
            raise GeminiError("GEMINI_API_KEY is not configured")
        with genai.Client(api_key=os.environ["GEMINI_API_KEY"]) as client:
            return client.models.generate_content(model=model, contents=parts,
                                                   config=types.GenerateContentConfig(**config))

    def _log(self, record):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.records.append(record)
        with (self.cache_dir / "calls.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(_safe(_canonical(record)) + "\n")

    def _raw(self, parts, *, role, tag, config):
        model = model_for(role)
        key = hashlib.sha256(_canonical([model, config, parts]).encode()).hexdigest()
        path = self.cache_dir / (key + ".json")
        start = time.monotonic()
        record = {"model": model, "role": role, "tag": _safe(tag), "key": key, "cached": False,
                  "served_by": model, "time": time.time(), "latency": 0.0, "attempts": 0}
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            record.update(cached=True, served_by=saved["served_by"], latency=time.monotonic() - start)
            self._log(record)
            return saved["text"], record
        if os.getenv("R2P_CACHE_ONLY") == "1":
            record["error"] = "CacheMiss"
            self._log(record)
            raise CacheMiss(f"Cache-only miss for {model}, tag={_safe(tag)}, key={key}")
        for attempt in range(self.retries + 1):
            record["attempts"] += 1
            try:
                reply = self._request(model, parts, config)
                if isinstance(reply, str):
                    result, served_by = reply, model
                elif isinstance(reply, tuple):
                    result, served_by = reply
                else:
                    result = reply.text
                    served_by = getattr(reply, "model_version", None) or model
                if not isinstance(result, str) or not result.strip():
                    raise GeminiError("Gemini returned no text")
                result = _safe(result)
                record.update(served_by=_safe(served_by), latency=time.monotonic() - start)
                atomic_json(path, {"text": result, "served_by": record["served_by"], "model": model})
                self._log(record)
                return result, record
            except Exception as exc:
                daily = _status(exc) == 429 and _daily_quota(exc)
                if _status(exc) == 429 and not daily and attempt < self.retries:
                    self.sleep(min(30.0, 2.0 ** attempt))
                    continue
                record.update(error="QuotaError" if daily else type(exc).__name__, latency=time.monotonic() - start)
                self._log(record)
                if daily and role == "main" and os.getenv("R2P_ALLOW_FALLBACK") == "1":
                    text, fallback = self._raw(parts, role="small", tag=tag + ":fallback", config=config)
                    return text, {**fallback, "fallback_from": model}
                if daily:
                    raise QuotaError(f"Daily/zero-limit quota exhausted for {model}; no automatic role fallback") from None
                raise GeminiError(f"Request failed for {model}: {_safe(str(exc))}") from None
        raise AssertionError("unreachable")

    def call(self, parts, *, role="main", tag, schema=None, thinking=None):
        if isinstance(parts, str):
            parts = [parts]
        else:
            parts = list(parts)
        budget = int(os.getenv("R2P_THINKING", "2048")) if thinking is None else int(thinking)
        if not 0 <= budget <= 32768:
            raise ValueError("thinking budget must be between 0 and 32768 tokens")
        config = {"temperature": 0, "thinking_config": {"thinking_budget": budget}}
        if schema is not None:
            json_schema = schema.model_json_schema() if hasattr(schema, "model_json_schema") else schema
            config.update(response_mime_type="application/json", response_json_schema=json_schema)
        text, record = self._raw(parts, role=role, tag=tag, config=config)
        if schema is None:
            return text, record
        try:
            errors = schema_errors(parse_json(text), schema)
        except JSONError as exc:
            errors = [str(exc)]
        if not errors:
            return text, record
        # Cache the original first: reruns reuse both it and this deterministic repair.
        repair = ["Repair the JSON syntax/schema only. Do not invent new observations. Return JSON.\n"
                  + "Errors: " + json.dumps(errors) + "\nOriginal reply:\n" + text]
        repaired, repair_record = self._raw(repair, role="small", tag=tag + ":json_repair", config=config)
        errors = schema_errors(parse_json(repaired), schema)
        if errors:
            raise JSONError("JSON repair failed: " + "; ".join(errors))
        return repaired, {**record, "repair": repair_record, "answer_served_by": repair_record["served_by"]}

    __call__ = call


class FakeGemini(Gemini):
    """Scripted real client/cache behavior, with responses, exceptions or a responder."""
    def __init__(self, responses=(), *, cache_dir=None, responder=None, sleep=lambda _: None, retries=3):
        self.responses = deque(responses)
        self.requests = []
        self.responder = responder
        self._temporary = tempfile.TemporaryDirectory(prefix="r2p-gemini-") if cache_dir is None else None
        super().__init__(cache_dir or self._temporary.name, transport=self._fake, sleep=sleep, retries=retries)

    def _fake(self, **request):
        self.requests.append(request)
        if self.responder:
            value = self.responder(**request)
        elif self.responses:
            value = self.responses.popleft()
        else:
            raise AssertionError("FakeGemini exhausted; no network request was made")
        if isinstance(value, Exception):
            raise value
        return json.dumps(value) if isinstance(value, (dict, list)) else value


def call(parts, *, role="main", tag, schema=None, thinking=None):
    return Gemini().call(parts, role=role, tag=tag, schema=schema, thinking=thinking)
