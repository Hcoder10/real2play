"""Role-aware Gemini calls with an offline, content-addressed cache.

Only this module talks to google-genai. Images are BGR (OpenCV convention).
The transport is injectable; FakeGemini never creates a Google client.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
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


class RequestLimit(GeminiError):
    """The caller's explicit live-request budget has been spent."""


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
    model = os.getenv("R2P_MODEL_" + role.upper(), "gemini-4-argon")
    if role == "small" and model == "gemini-3.8-flash":
        return model if os.getenv("R2P_ALLOW_FLASH") == "1" else "gemini-4-argon"
    if model != "gemini-4-argon":
        raise ValueError("Only gemini-4-argon is permitted (approved SMALL calls may use gemini-3.8-flash)")
    return model


def thinking_level(tag, thinking=None):
    """Translate legacy numeric budgets at the boundary; never send token budgets."""
    value = thinking if thinking is not None else os.getenv("R2P_THINKING")
    if value is not None:
        if str(value) in ("low", "medium", "high"):
            return str(value)
        try:
            budget = int(value)
        except (TypeError, ValueError):
            raise ValueError("thinking must be low, medium, high or a legacy 0..32768 budget") from None
        if not 0 <= budget <= 32768:
            raise ValueError("thinking budget must be between 0 and 32768 tokens")
        return "low" if budget <= 1024 else "medium" if budget <= 4096 else "high"
    if any(word in tag for word in ("active_choose", "active_patch", "experiment", "patch_design", "build_game", "author_task")):
        return "high"
    if any(word in tag for word in ("point", "vcal_tips", "ttt:caps", "detect", "perception_boxes", "task_from_video", "labeling", "json_repair")):
        return "low"
    return "medium"


def interaction_input(parts):
    """Keep the public image/video helpers compatible; convert at the SDK boundary."""
    result = []
    for part in parts:
        if isinstance(part, str):
            result.append({"type": "text", "text": part})
        elif isinstance(part, dict) and "inline_data" in part:
            data = part["inline_data"]
            mime = data["mime_type"]
            kind = mime.split("/")[0]
            if kind not in ("image", "video"):
                raise ValueError("Only inline images and videos are supported")
            encoded = data["data"]
            if isinstance(encoded, bytes):
                encoded = base64.b64encode(encoded).decode("ascii")
            result.append({"type": kind, "mime_type": mime, "data": encoded})
        elif isinstance(part, dict) and part.get("type") in ("text", "image", "video"):
            result.append(part)
        else:
            raise ValueError("Expected text, image_part, video_part or Interactions content")
    return result


def field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def plain(value):
    return value.model_dump(exclude_none=True, by_alias=True) if hasattr(value, "model_dump") else value


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
        if str(value).isdigit():
            return int(value)
    match = re.search(r"\b(408|429|5\d\d)\b", str(exc))
    return int(match[1]) if match else (429 if "RESOURCE_EXHAUSTED" in str(exc) else None)


def _transient(exc):
    if isinstance(exc, GeminiError):
        return False  # A lifecycle failure must never launch the background job again.
    status = _status(exc)
    if status in (408, 429) or status is not None and 500 <= status <= 599:
        return True
    # The SDK uses httpx; importing it is unnecessary for injected/offline clients.
    return isinstance(exc, (TimeoutError, ConnectionError)) or type(exc).__name__ in (
        "ReadTimeout", "ConnectTimeout", "WriteTimeout", "PoolTimeout", "ConnectError", "ReadError")


class Gemini:
    def __init__(self, cache_dir=None, *, transport=None, sleep=time.sleep, retries=3,
                 timeout_s=None, max_requests=None, poll_interval_s=5, clock=time.monotonic):
        self.cache_dir = Path(cache_dir or os.getenv("R2P_GEMINI_CACHE", "cache/gemini"))
        self.transport = transport
        self.sleep = sleep
        self.retries = retries
        self.timeout_s = float(timeout_s if timeout_s is not None else os.getenv("R2P_GEMINI_TIMEOUT_S", "1200"))
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0 or retries < 0:
            raise ValueError("Timeout must be positive and retries nonnegative")
        self.max_requests = max_requests
        self.requests_used = 0
        self.records = []
        self.poll_interval_s = poll_interval_s
        self.clock = clock
        if not math.isfinite(poll_interval_s) or poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")

    def _reserve(self):
        if self.max_requests is not None and self.requests_used >= self.max_requests:
            raise RequestLimit(f"Live request budget exhausted ({self.max_requests})")
        self.requests_used += 1

    def _client(self):
        from google import genai
        from google.genai import types
        if not os.getenv("GEMINI_API_KEY"):
            raise GeminiError("GEMINI_API_KEY is not configured")
        options = types.HttpOptions(timeout=round(self.timeout_s * 1000),
                                    retry_options=types.HttpRetryOptions(attempts=0))
        return genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=options)

    def _poll(self, client, reply, *, tag, deadline):
        interaction_id = field(reply, "id")
        while field(reply, "status") in ("queued", "in_progress"):
            if not interaction_id:
                raise GeminiError("Background interaction has no id")
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise GeminiError(f"Interaction timed out; resume id={interaction_id}")
            self.sleep(min(self.poll_interval_s, remaining))
            if self.clock() >= deadline:
                raise GeminiError(f"Interaction timed out; resume id={interaction_id}")
            for attempt in range(self.retries + 1):
                start = self.clock()
                record = {"api": "interactions", "operation": "get", "tag": tag,
                          "interaction_id": interaction_id, "served_by": None, "cached": False}
                try:
                    self._reserve()
                    reply = client.interactions.get(id=interaction_id, timeout=max(.001, deadline - self.clock()))
                    record.update(status=field(reply, "status"), served_by=field(reply, "model"),
                                  usage=plain(field(reply, "usage")), latency=self.clock() - start)
                    self._log(record)
                    break
                except Exception as exc:
                    record.update(error=type(exc).__name__, latency=self.clock() - start)
                    self._log(record)
                    if _status(exc) == 429 and _daily_quota(exc):
                        raise QuotaError(f"Polling quota exhausted; resume id={interaction_id}") from None
                    if _transient(exc) and attempt < self.retries and self.clock() < deadline:
                        self.sleep(min(2 ** attempt, max(0, deadline - self.clock())))
                        continue
                    if isinstance(exc, RequestLimit):
                        raise
                    raise GeminiError(f"Polling failed; resume id={interaction_id}: {_safe(exc)}") from None
        status = field(reply, "status")
        if status and status != "completed":
            raise GeminiError(f"Interaction {interaction_id} ended with status={status}: {_safe(field(reply, 'errors', []))}")
        return reply

    def _request(self, model, parts, config):
        if self.transport is not None:
            return self.transport(model=model, parts=parts, config=config)
        deadline = self.clock() + self.timeout_s
        with self._client() as client:
            reply = client.interactions.create(model=model, input=interaction_input(parts),
                                               **config, timeout=self.timeout_s)
            tag = getattr(self, "_active_tag", "model:background")
            if field(reply, "status") in ("queued", "in_progress"):
                self._log({"api": "interactions", "operation": "started", "tag": tag,
                           "interaction_id": field(reply, "id"), "served_by": field(reply, "model"),
                           "cached": False, "latency": 0, "status": field(reply, "status")})
            return self._poll(client, reply, tag=tag, deadline=deadline)

    def _log(self, record):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.records.append(record)
        with (self.cache_dir / "calls.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(_safe(_canonical(record)) + "\n")

    def _raw(self, parts, *, role, tag, config):
        model = model_for(role)
        if model == "gemini-3.8-flash":
            logging.getLogger(__name__).warning("R2P_ALLOW_FLASH=1: gemini-3.8-flash call site %s", _safe(tag))
        key = hashlib.sha256(_canonical(["interactions/v1", model, config, parts]).encode()).hexdigest()
        path = self.cache_dir / (key + ".json")
        start = time.monotonic()
        record = {"model": model, "role": role, "tag": _safe(tag), "key": key, "cached": False,
                  "served_by": None, "time": time.time(), "latency": 0.0, "attempts": 0,
                  "attempt_details": [], "api": "interactions", "config": config, "usage": None}
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            record.update(cached=True, served_by=saved["served_by"], usage=saved.get("usage"), latency=time.monotonic() - start)
            self._log(record)
            return saved["text"], record
        if os.getenv("R2P_CACHE_ONLY") == "1":
            record["error"] = "CacheMiss"
            self._log(record)
            raise CacheMiss(f"Cache-only miss for {model}, tag={_safe(tag)}, key={key}")
        for attempt in range(self.retries + 1):
            if self.max_requests is not None and self.requests_used >= self.max_requests:
                record.update(error="RequestLimit", latency=time.monotonic() - start)
                self._log(record)
                raise RequestLimit(f"Live request budget exhausted ({self.max_requests})")
            self.requests_used += 1
            record["attempts"] += 1
            attempt_start = time.monotonic()
            try:
                self._active_tag = _safe(tag)
                reply = self._request(model, parts, config)
                if isinstance(reply, str):
                    result, served_by = reply, model
                elif isinstance(reply, tuple):
                    result, served_by = reply
                else:
                    result = field(reply, "output_text")
                    served_by = field(reply, "model") or model
                    record.update(usage=plain(field(reply, "usage")), interaction_id=field(reply, "id"))
                if not isinstance(result, str) or not result.strip():
                    raise GeminiError("Gemini returned no text")
                result = _safe(result)
                record["attempt_details"].append({"latency": time.monotonic() - attempt_start,
                                                  "served_by": _safe(served_by), "status": "ok"})
                record.update(served_by=_safe(served_by), latency=time.monotonic() - start)
                atomic_json(path, {"text": result, "served_by": record["served_by"], "model": model, "usage": record["usage"]})
                self._log(record)
                return result, record
            except Exception as exc:
                daily = _status(exc) == 429 and _daily_quota(exc)
                record["attempt_details"].append({"latency": time.monotonic() - attempt_start,
                                                  "status": _status(exc), "error": type(exc).__name__,
                                                  "served_by": None})
                if _transient(exc) and not daily and attempt < self.retries:
                    self.sleep(min(30.0, 2.0 ** attempt))
                    continue
                record.update(error="QuotaError" if daily else type(exc).__name__, latency=time.monotonic() - start)
                self._log(record)
                if daily and role == "main" and os.getenv("R2P_ALLOW_FALLBACK") == "1" and model_for("small") != model:
                    text, fallback = self._raw(parts, role="small", tag=tag + ":fallback", config=config)
                    return text, {**fallback, "fallback_from": model}
                if daily:
                    raise QuotaError(f"Daily/zero-limit quota exhausted for {model}; no automatic role fallback") from None
                if isinstance(exc, (RequestLimit, QuotaError)):
                    raise
                raise GeminiError(f"Request failed for {model}: {_safe(str(exc))}") from None
        raise AssertionError("unreachable")

    def call(self, parts, *, role="main", tag, schema=None, thinking=None, background=False, max_output_tokens=8192):
        if isinstance(parts, str):
            parts = [parts]
        else:
            parts = list(parts)
        if not isinstance(max_output_tokens, int) or isinstance(max_output_tokens, bool) or max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        config = {"generation_config": {"thinking_level": thinking_level(tag, thinking),
                                         "max_output_tokens": max_output_tokens}, "background": bool(background)}
        if schema is not None:
            json_schema = schema.model_json_schema() if hasattr(schema, "model_json_schema") else schema
            config["response_format"] = {"type": "text", "mime_type": "application/json", "schema": json_schema}
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
        repair_config = {**config, "generation_config": {**config["generation_config"], "thinking_level": "low"}}
        repaired, repair_record = self._raw(repair, role="main", tag=tag + ":json_repair", config=repair_config)
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
            # Existing offline pipeline responders inspect the old schema key.
            # This alias is fake-only; it never enters the SDK body or cache key.
            config = request["config"]
            if "response_format" in config:
                config = {**config, "response_json_schema": config["response_format"]["schema"]}
            value = self.responder(**{**request, "config": config})
        elif self.responses:
            value = self.responses.popleft()
        else:
            raise AssertionError("FakeGemini exhausted; no network request was made")
        if isinstance(value, Exception):
            raise value
        return json.dumps(value) if isinstance(value, (dict, list)) else value


def call(parts, *, role="main", tag, schema=None, thinking=None, background=False, max_output_tokens=8192):
    return Gemini().call(parts, role=role, tag=tag, schema=schema, thinking=thinking,
                         background=background, max_output_tokens=max_output_tokens)
