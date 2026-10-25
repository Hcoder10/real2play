# Agent D: Argon brain

All coordinates are spec-frame metres; all joint targets are absolute MJCF radians.
The entry points bootstrap the repo imports, so they work from another working directory too.

## Offline acceptance

```powershell
.venv/Scripts/python.exe -m pytest -q tests/argon
.venv/Scripts/python.exe scripts/argon/tictactoe.py --dry-run --games 2
```

The dry run renders C2's CPU synthetic bench and reads its board through an explicit oracle.
It exercises game rules, projected/warped images, validation and minimax. It does not claim to
evaluate a placement policy or call Gemini. `--oracle-only` works without synthetic rendering.
The exhaustive minimax tests explore every human continuation for both possible robot marks.

## Gemini

`r2p.gemini.call(parts, role="main", tag=..., schema=...)` returns `(text, record)`.
MAIN defaults to `gemini-4-argon`, SMALL to `gemini-3.8-flash`; override with
`R2P_MODEL_MAIN` / `R2P_MODEL_SMALL`. The installed `google-genai` extra is the only network client.
The [official SDK reference](https://googleapis.github.io/python-genai/) describes its request types.

Set `GEMINI_API_KEY` in the environment for an authorized live run. It is never included in cache
records or call logs. `R2P_THINKING` defaults to 2048 and must be 0..32768. `R2P_CACHE_ONLY=1` makes
zero requests and raises `CacheMiss` for any uncached request, including JSON repair. Cache entries
are keyed on model, config and all part bytes under `cache/gemini`; `R2P_GEMINI_CACHE` changes that
directory. `calls.jsonl` records model, actual `served_by`, attempts, latency and cache state.

Per-day and zero-limit 429s raise `QuotaError`. Per-minute errors use bounded retries; the word
“billing” is never a quota classification signal. Only `R2P_ALLOW_FALLBACK=1` permits MAIN to fall
back to SMALL. JSON repair uses SMALL on the cached original reply, never repeats the expensive
vision prompt, and records both answering models. `FakeGemini` exercises this same client/cache
path with scripted responses or an injected responder.

## Tic-tac-toe

```powershell
.venv/Scripts/python.exe scripts/argon/tictactoe.py --frame-only frame.jpg --camera camera.json --history history.json
.venv/Scripts/python.exe scripts/argon/tictactoe.py --real --policy exports/place_on_cell --supply-xyz 0.18 -0.12 0.016 --piece-size 0.032
```

`--robot-desc` and `--human-desc` describe the pieces. `--piece-size` is metres. History is either
`[{"cell":"r0c0","mark":"X"}, ...]` or `{"board":["X","", ...]}`. Earlier marks cannot change.
One MAIN call receives the annotated real frame, perspective-warped grid, descriptions and history;
one validation re-ask is permitted. A consistent board permits perfect local minimax fallback. If
the current human move cannot be established, the live loop requires its operator confirmation.

Live mode uses the already-running C1 daemon, a verified joint map, measured supply coordinates,
the calibrated camera and a `place_on_cell` export. Each placement is its own C1 `Session`, starts
from the export's pose and records a trace plus camera calibration. Failed placements stop the game.
The known supply projection seeds perception; no inferred supply coordinate is used.
`out/ttt/status.json` is atomically updated with `argon_says`, move, board, model and status.

## Task authoring and scene fitting

```powershell
.venv/Scripts/python.exe scripts/argon/task_from_video.py runs/robot/attempt --out-live out/live/example --iters 6
.venv/Scripts/python.exe scripts/argon/task_from_video.py episode.mp4 --instruction "put the cube in the bowl" --no-twin
.venv/Scripts/python.exe scripts/argon/run_twin.py runs/robot/attempt --task cube_in_bowl --out out/live/example2
```

A recorded session contains `trace.jsonl` (`r2p_trace/1`), `frames/`, and `camera.json`. Tick `frame`
references are authoritative. Without them, use one frame per tick or explicit `frame_times` /
`frame_fps` in metadata; mismatched video/joint clocks are never guessed. The loader also accepts
bare MP4/MOV/WebM/AVI videos and images (a single image requires an instruction).

Task inference receives the whole video, first/last frames, a 5 Hz joint table and measured anchor
facts. It validates through `r2p.tasks.validate`, re-asks at most twice with exact errors, fills
randomization locally and pins badly mismatched grasp width. It writes `tasks/<name>/task.json`.
A calibrated episode with at least two frames then enters the twin loop and writes `scene.json`.
A bare uncalibrated video produces a task with `scene_status=requires_calibrated_camera`; metric
geometry is not claimed without calibration. A video without joints cannot validate robot physics.

The loop uses one SMALL detection for the critic frame set, local Lab appearance masks and
calibrated silhouette fitting. `TaskTwin` replays **recorded q_target**, and C2's CPU mesh renderer
renders its joint/object state through the real camera. No CUDA, GPU lease or graphics context is
needed. The first and subsequent critic images never include held-out evaluation frames. Local
metrics include per-object IoU, centroid pixels, trajectory pixels, camera keypoints when supplied,
and grasp/release anchors. A missing camera-keypoint measurement stays null; calibration is locked.

Gripper commands define close/release times and the most-closed empty reference. Settled measured
jaw gaps establish grasp width; an empty close is not a grasp. `--anchors auto` enforces only
image-consistent anchors (within 25 mm); `hard` forces enforcement and `soft` reports them only.
Widths are constrained to ±20% of measured gap. Candidate poses remain scoreable so regressions
are rejected, not hidden. The keep rule requires gain >0.003, per-object IoU no more than 0.10 below
its observed best, container loss no more than 0.02 without enforced anchors, camera error no worse,
and, with enforced anchors, grasp error no worse and release inside the container. Two unsuccessful
iterations stop the loop. Invalid patches are rejected. Quota/cache-only stops retain the best scene.

`out/live/<run>/` contains atomic `r2s_live/1` `run.json`, each `iter_N.json`, candidate scene JSON,
1280×480 `iter_N_overlay_J.jpg`, `iter_N_physics.jpg`, and `scene_best.json`. Existing run directories
are never overwritten. The dashboard's per-model request counts include retries and repairs.

## Labeling and system identification

```powershell
.venv/Scripts/python.exe scripts/argon/label_trace.py episode.jsonl labeled.jsonl --task cube_in_bowl
.venv/Scripts/python.exe scripts/argon/sysid.py plan residuals.json
.venv/Scripts/python.exe scripts/argon/sysid.py fit train1.jsonl train2.jsonl --heldout heldout.jsonl --physics so101_bench
```

`segment(trace, task)` returns contiguous `[t0,t1,command]` intervals from existing command labels,
gripper commands and motion phases. Optional `--refine` uses one MAIN call; invented vocabulary,
gaps and overlaps are rejected. `label_trace` writes the shared trace format, including meta.commands.

The planner prioritizes residual/tolerance, proposes bounded absolute joint excitation and resettable
push/grasp/drop probes, and never executes hardware. `fit_parameters(train, physics, heldout=...)`
fits bounded servo gains/friction against replay trajectories, searches integer latency, and fits
runner `sag_gain` separately from static holds because TaskTwin derives sag from finite stiffness.
Default object friction is applied to replayed dynamic objects; per-object overrides can be fitted
with parameter paths such as `objects.cube.friction` when present in the supplied physics dictionary.
Focused `Parameter(path, low, high)` sets avoid fitting parameters the experiment does not excite.

Whole training/held-out traces must differ. Rank-deficient fits, unconverged optimizers, absent evidence,
and insufficient static holds fail certification. Held-out gates are jaw RMS <2 mm, maximum object
endpoint error <5 mm, physical opposing-jaw grasp agreement ≥95%, and timing error <1 tick. Grasp
success requires opposing physical jaw contacts across three ticks. `--output NEW.json` publishes
only a passing new version; it never overwrites an incumbent. Real benchmark accuracy and live Argon
behavior remain unmeasured until recorded hardware trials and authorized model calls are available.
