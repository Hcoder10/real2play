# Real2Play

Show Gemini 4 Argon your robot. It experiments on the real arm until the twin is exact, turns your task into a game the
world plays (Roblox), and trains the robot on that play. Gemini plans; the policy moves.

Design: [docs/SPEC.md](docs/SPEC.md). Agent rules: [AGENTS.md](AGENTS.md).

## How Gemini 4 Argon is used

All reasoning and perception default to **`gemini-4-argon`**, through the Interactions API
in `r2p/gemini.py` (google-genai 2.33.0+). This is the current owner policy, superseding the
older default-SMALL wording in the design notes. `R2P_MODEL_MAIN` accepts only Argon.
SMALL also resolves to Argon; using `gemini-3.8-flash` requires both
`R2P_MODEL_SMALL=gemini-3.8-flash` and explicit owner approval via `R2P_ALLOW_FLASH=1`.
Each Flash call warns with its call-site tag. No other models are accepted.

| Call site | Purpose | Role / model by default | Thinking level |
|---|---|---|---|
| `perception_boxes` (`r2p/perception/task_perception.py`) | Initial object boxes | SMALL / Argon | low |
| `real2sim:detect` (`r2p/real2sim/detect.py`) | Scene extraction | SMALL / Argon | low |
| `task_from_video:*` (`r2p/real2sim/task_video.py`) | Whole-video task extraction | MAIN / Argon | low |
| `visual:point` (`r2p/sysid/visual.py`), `vcal_tips` (`scripts/lead/visual_calib.py`) | Gripper endpoints | MAIN / Argon | low |
| `ttt:caps` (`scripts/lead/ttt_armd.py`) | Supply-cap localization | MAIN / Argon | low |
| `active:cap_pose` (`scripts/lead/active_live.py`) | Active experiment cap localization | MAIN / Argon | low |
| `labeling:refine` (`r2p/labeling.py`) | Trace skill segmentation | MAIN / Argon | low |
| `ttt:decide:*` (`r2p/ttt`) | Read board and choose move | MAIN / Argon | medium |
| `real2sim:critique:*` (`r2p/real2sim/loop.py`) | Compare real/rendered observations and propose bounded patch | MAIN / Argon | medium |
| `active_choose`, `active_patch` (`r2p/real2sim/active.py`) | Experiment and patch design | MAIN / Argon | high |
| `*:json_repair` (`r2p/gemini.py`) | One bounded schema repair | MAIN / Argon | low |
| `build_game`, `author_task` (`r2p/agents.py`, `scripts/argon/`) | Managed sandbox authoring and tests | Antigravity / Argon | High effort requested in prompt; agent API controls reasoning |

The model client sends `generation_config.thinking_level` and `max_output_tokens`, typed
image/video inputs, and `response_format={type: "text", mime_type: "application/json", schema: ...}`.
It reads `interaction.output_text`; long calls support `background=True` and polling by ID.
There are no temperature, top-p/top-k, token thinking-budget or explicit cached-content fields.
Legacy numeric `thinking` values map to low (0–1024), medium (1025–4096), high (4097–32768).
See the [Google Interactions API reference](https://ai.google.dev/api/interactions-api).

The disk cache hashes the API version, model, request config and all input bytes.
`R2P_CACHE_ONLY=1` makes zero requests, including repairs and managed-agent polling.
`calls.jsonl` records served model, latency, usage, cache state and attempts; SDK retries
are disabled, and application retries use bounded backoff. Daily/zero-limit quota raises
`QuotaError`; fallback is limited to low-effort extraction/pointing/repair calls and additionally
requires `R2P_ALLOW_FALLBACK=1` and approved Flash. Board, critique and experiment reasoning stay on Argon. HTTP timeout defaults to 20 minutes (`R2P_GEMINI_TIMEOUT_S`). Keys stay out of prompts,
caches and logs. Model caches use implicit service caching in addition to the local disk cache.

Our render → watch → critique → fix loop takes reality as its goal. Recorded commands replay
through the twin; the calibrated renderer produces comparison frames; Argon proposes changes;
local checks retain only candidates that satisfy the held-out keep rule. The active loop chooses
experiments to reduce uncertainty and separates fitting clips from hold-out experiments. A
repeated hold-out gate is model selection, not an independent real-robot transfer benchmark.

The managed game builder receives task, measured board, physics, camera and harness sources.
`antigravity-preview-09-2026` uses `agent_config={type: "antigravity", model: "gemini-4-argon"}`
in a remote environment, creates the game configuration and runs sandbox tests. Returned files
are applied in an isolated local workspace; all Luau suites and numerical trajectory parity
must pass. The task author similarly returns a task and positive/negative checker vectors,
which must pass the existing `r2p.tasks` checker and scene sampling locally. Neither stage
publishes. Numerical tests do not replace rendered-game review, Studio contact parity or
held-out real-world twin verification. Run instructions and evidence format are in
[scripts/argon/README.md](scripts/argon/README.md).

Sources are mounted as SDK inline files, and missing or changed immutable-source hashes fail
the stage. These reported hashes supplement independent local checks; they are not trusted
attestation. The first live managed run returned locally valid files but altered sandbox tests,
so its overall result was rejected. See [J's evidence report](scripts/argon/J_INTERACTIONS_REPORT.md).
