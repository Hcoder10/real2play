# Agent J: Interactions API, Argon-only models, Antigravity managed agents

Read docs/briefs/COMMON.md, AGENTS.md, docs/INTERFACES.md first. Repo C:\Users\sarta\real2play,
`.venv/Scripts/python.exe` (google-genai 2.33.0 installed; it has `client.interactions` and `client.agents`). Commit with
explicit paths only, small commits, pull before editing shared files (agent I is building r2p/real2sim/active.py in
parallel; the lead edits scripts/lead and config). Full suite before each commit:
`.venv/Scripts/python.exe -m pytest -q tests/core tests/argon tests/perception tests/robot tests/app tests/sim tests/policy`.
GEMINI_API_KEY is in the gitignored .env (auto-loaded by r2p/gemini.py). NEVER print it. Live budget: 25 requests.
Deadline 13:45.

## Owner's rules (binding)
- Models: gemini-4-argon is THE model for everything. gemini-3.8-flash may be used only for small tasks AND only when
  the owner approved it: make the SMALL role resolve to gemini-4-argon by default; flash only when
  `R2P_ALLOW_FLASH=1` is set (log a warning naming the call site). No other models (TTS excepted, not needed now).
- The hackathon guide (Gemini 4 Argon builder guide) says: use the Interactions API
  (`client.interactions.create(model="gemini-4-argon", input=..., generation_config={"thinking_level": "low"|"medium"|
  "high", "max_output_tokens": ...})`, `interaction.output_text`, `background=True` + poll `client.interactions.get(id=...)`
  for long jobs); remove temperature/top_p/top_k and thinking_budget (deprecated for argon); HTTP timeouts 15-20 min for
  "high"; implicit caching (no cached_content). Managed agent: `client.interactions.create(agent="antigravity-preview-
  09-2026", agent_config={"type": "antigravity", "model": "gemini-4-argon"}, input=..., environment="remote")` runs
  Argon in a Google-hosted Linux sandbox that writes code, runs tests, manages files. Secrets: never put the key or any
  token in a sandbox input.
- Problem statement we answer (Problem Statement Two, "More Dimensions"): Argon is SOTA on long video understanding and
  generates 3D games/worlds in one pass; a single pass is a first draft: build a system that renders what it made,
  watches the result, critiques it against the goal, and fixes it in a loop. Ours: the goal is REALITY (the real robot
  video); Argon renders the twin, watches real vs sim video, critiques, patches, keeps what scores better on held-out
  clips; then Argon generates the Roblox game from the verified twin and verifies it the same way.

## Build
1. r2p/gemini.py on the Interactions API. Discover the exact SDK surface by introspection in the venv
   (`help(client.interactions.create)`, types) rather than guessing. Keep: roles, disk cache keyed on request,
   calls.jsonl logging with served_by/latency/usage, R2P_CACHE_ONLY, QuotaError, retries/backoff on 5xx/429, JSON schema
   output (find how the Interactions API takes a response schema / mime type; if it cannot, request JSON and validate
   + repair with argon), image and video parts (find the Interactions input format for inline images/video). Map the old
   `thinking` arg to thinking_level per call site (pointing/extraction: low; board decisions, critique: medium; patch
   design, experiment design, game generation: high). One live smoke call per mode (text, image+schema, background).
2. r2p/agents.py: a small Antigravity managed-agent wrapper (start, poll, collect output_text and any files/artifacts
   the API returns; timeouts; log to calls.jsonl). Find from the SDK how files come back (or ask the agent to return a
   single JSON with {"files": {path: content}, "tests": ..., "report": ...} and parse that).
3. Use managed agents where they genuinely help (each must be a real step of the pipeline with a pass/fail check):
   a. GAME BUILDER (the main one): given the verified twin (task.json, config/physics/*.json, measured board, camera)
      and the Roblox harness sources (roblox/ - pass the relevant files inline; the context is 1M tokens), the
      Antigravity agent generates/adapts the game level config + any Luau for that task, runs the harness's Luau unit
      tests and parity checks in its sandbox (it can install luau/lune or a Lua 5.x there; or port the check to
      Python if needed), iterates until they pass, and returns the files; we apply them locally behind our own
      verification (our Luau tests: see roblox/README.md / roblox/ACCEPTANCE.md) and never auto-publish.
   b. TASK + CHECKER AUTHOR: from a natural-language goal + a real demo clip summary, the agent writes a new task JSON
      and test vectors and runs r2p.tasks checks in the sandbox (pass the needed r2p/tasks sources inline).
   Wire them as pipeline stages/CLI: scripts/argon/build_game.py, scripts/argon/author_task.py, each with --dry-run
   (fake agent) for tests. Do one real managed-agent run of (a) and record time, result, and what passed.
4. Update docs (README section "How Gemini 4 Argon is used": every call site, model, thinking level, managed agents,
   the render-watch-critique-fix loops) and tests (fakes only).
Report: what changed, live smoke results (latency per mode), the managed-agent run result, commits.
