# Agent R handoff — 25 October 2026

Implementation commit: **`37388e7b610a8a25d722cbb787618844b3258238`**.
Completed and verified before the 14:35 deadline. Only `r2p/lab/`,
`scripts/argon/lab.py`, and `tests/lab/` were changed. No real arm motion or GPU
work was performed. Other agents' files were left untouched.

Implemented the loopback observation/plan server, guarded bench primitives,
dual-camera episode recording, editable static MuJoCo workspace, timestamped PD
replay, side-by-side videos and residuals, immutable object observations, sealed
held-out scoring, high-thinking Argon loop, managed static-asset build adapter,
hidden-truth dry backend, and CLI. See [README.md](README.md) for the protocol,
configuration, safety rules and renderer limitations.

Acceptance command:

```powershell
.venv/Scripts/python.exe -m pytest tests/lab -q
```

Final run used `--basetemp runs/lab/R-release-acceptance` to retain the evidence:
**45 passed in 57.04 seconds**. `compileall`, CLI help and staged whitespace checks
also passed. Tests use fake model transports and injected/MuJoCo robot backends.

The five-turn fake-agent integration test exercises collection, two untouched
held-out episodes, static parameter editing, training replay, final scoring and
completion. Its canned fit is a test oracle for a synthetic fixture, not live
Argon identification. In this no-object motion fixture:

| Held-out metric | Baseline | Candidate |
| --- | ---: | ---: |
| Joint RMS, degrees | 0.2398 | 0.0878 |
| Jaw path RMS, mm | 2.5405 | 0.9688 |
| Image similarity | 0.94750 | 0.99721 |

Final-pose object error remains `null` in that fixture; no object accuracy is
claimed. Separate tests require independent object labels, reject missing or
mutable held-out evidence, and validate physical collision geometry. Identical
physics replay reproduces recorded joint and jaw trajectories exactly, including
an irregular timestamp test. Other checks cover speed caps, live height aborts,
vertical grid access, contact relief, sustained gripper stall, sheet movement,
owner approval, daemon refusals, HTTP origin rejection, path traversal, collision
mask isolation, XML includes, asset rollback, server-side model state, malformed
responses, and managed-bundle validation.

Retained demonstration artifacts:

- Live Argon dry turn: `runs/lab/R-argon-dry-03/ledger.jsonl` and `status.json`.
  Annotated reference images: `obs/scene.jpg`, `obs/wrist.jpg`; candidate images:
  `previews/scene.jpg`, `previews/wrist.jpg`.
- Full fake-agent loop:
  `runs/lab/R-release-acceptance/test_full_fake_argon_experimen0/run/`.
  Its `status.json` contains baseline/candidate metrics and the replay path.
- Training comparison movie:
  `runs/lab/R-release-acceptance/test_full_fake_argon_experimen0/run/replays/1-7e0b52e09f/side_by_side.mp4`.
  Episode folders preserve both raw camera movies, JPGs, telemetry and commands.

Successful live dry command, with no hardware connected to this run:

```powershell
.venv/Scripts/python.exe scripts/argon/lab.py --task tasks/cube_in_bowl/task.json --run R-argon-dry-03 --real-mock --max-turns 1 --approve auto-sim --request-budget 1 --timeout 180 --port 0
```

The response was served by **gemini-4-argon**, uncached, at high thinking level;
API latency was **18.188 seconds**, with an Interactions ID recorded in the
ledger. The first schema-based attempt omitted its plan/tool arguments and was
rejected. A subsequent attempt received `content_blocked` from the API. The
successful third attempt used local JSON validation and explicit simulation-only
scope. No failed attempt executed physical motion.

Live transcript excerpt:

```text
# Plan:
1. Inspect `scene.xml` and `physics.json` and render initial simulation views (`scene` and `wrist`).

# Thought: We need to align the editable simulation (`scene.xml` and `physics.json`)
with the reference 'real' environment both visually and physically for the `cube_in_bowl` task.
```

Its tool calls, all successful:

```json
[
  {"tool":"read_sim","args":{"path":"scene.xml"}},
  {"tool":"read_sim","args":{"path":"physics.json"}},
  {"tool":"sim_render","args":{"camera":"scene"}},
  {"tool":"sim_render","args":{"camera":"wrist"}}
]
```

The one-turn live run correctly ended **incomplete / max_turns**, with `done:false`.
It demonstrates model/tool integration, not convergence.

Exact real-bench command (existing repository daemon and both camera servers
must be running; the owner sees and approves each proposed real plan):

```powershell
.venv/Scripts/python.exe scripts/argon/lab.py --task tasks/place_on_cell/task.json --run bench-argon-lab-R --max-turns 20 --approve ask --request-budget 40
```

Not performed: physical bench execution, real twin convergence, policy transfer,
and a live Antigravity build. No measured wrist calibration exists in the repo:
raw wrist recording works, but real metric wrist deprojection and `look_at`
explicitly refuse until the mount/intrinsic calibration described in the README
is supplied. CPU rendering is an approximate mesh renderer, not photorealistic.
