# Agent F: orchestration and dashboard

Run from the repository root with the shared Python environment:

```powershell
.venv/Scripts/python.exe -m pytest -q tests/app
.venv/Scripts/python.exe dashboard/server.py --mock
.venv/Scripts/python.exe dashboard/shots.py --out dashboard/shots/mock-1920x1080.png
```

The dashboard is at **http://127.0.0.1:8800**, with stage history at **/runs**.
`--mock` is an explicitly labelled illustration; its values are not experiment results.
The screenshot tool uses real wall-clock painting and the Chrome DevTools protocol,
with an isolated Chrome/Edge profile. It never sets a virtual-time budget. Its adjacent
JSON records viewport dimensions and JavaScript exceptions. `--cdp-url` can attach
to an existing local debugging endpoint. `--headful` is optional.

## Pipeline

```powershell
# Plan: stops before arm steps and prints their exact bench commands.
.venv/Scripts/python.exe -m r2p run --task cube_in_bowl

# Synthetic rehearsal. Model requests remain cache-only unless explicitly enabled.
.venv/Scripts/python.exe -m r2p run --task cube_in_bowl --dry-run --rounds 2 --device cpu

# Start with a recorded attempt; choose an existing incumbent export explicitly.
.venv/Scripts/python.exe -m r2p run --task cube_in_bowl --video runs/recorded/attempt_01 --incumbent exports/cube_in_bowl_v1

# Infer a task from a recorded episode, then use the same pipeline.
.venv/Scripts/python.exe -m r2p run --task "put the cube in the bowl" --video runs/recorded/attempt_01 --dry-run

.venv/Scripts/python.exe -m r2p status
.venv/Scripts/python.exe -m r2p status cube_in_bowl --json
.venv/Scripts/python.exe -m r2p run --task cube_in_bowl --dry-run --force r1:twin
```

Stages are `session → twin → close → data → label → train → eval → gate → deploy`.
`--force rN:sysid` aliases `close`. `--no-cascade` preserves valid downstream
artifacts for that forced invocation. Changed model/options or external inputs still
invalidate their caches. Output artifacts are hashed, so deletion or modification
invalidates a cached stage. A process lock prevents simultaneous writers; JSON is
flushed and atomically replaced, including bounded Windows sharing-violation retries.

Manifests use `r2p_run/1` at `runs/<task>/manifest.json`. Synthetic rehearsals use
`runs/<task>-dry/manifest.json`, carry `synthetic: true`, and never overwrite a real run.
Each stage records status, attempts, dependency fingerprints, output hashes, elapsed
time, and its result or clear BLOCKED reason. A missing component, Gemini cache miss,
daily quota exhaustion, missing lease, or missing incumbent is **BLOCKED** (exit 2).
Rerun the same command after resolving it; completed stages are reused.

Round 2 consumes round 1's recorded deployment attempt. Failed task attempts are
still useful evidence and are accepted when the runner records a valid trace and frames.
No result is invented for a failed component or a missing recorded attempt.

The initial `close` stage compares the baseline replay and proposes new experiments.
It does **not** identify physics from a single attempt or claim the twin is exact.
From round 2 onward, prior attempts train the fit and the current attempt is held out.
Only a passing sysid gate updates the versioned physics used for expert traces and
evaluation. Positions in focus scenes come from recorded perception; twin object
physics are retained only when the replay validation passes. Parameters and proposed
experiments stay under the run directory, never overwriting shared bench configuration.

Data mixes prior/focus expert traces and task-matched `roblox/out/traces` episodes.
Matching uses metadata, not filenames. Previous rounds' accepted trace indices are
included. Labelling writes separate command-labelled copies and preserves q_target.

Training requires a landlord receipt at `runs/lease.json` (or `--lease`), including
`lease_id`, the returned environment, and `vram_gb`. It verifies the active lease
against the live ledger before launch, applies the returned CUDA visibility/compute
environment, and derives/enforces the PyTorch VRAM fraction. The BLOCKED message
prints the exact `landlord_overview` / `rent_apartment` / rerun / `move_out` steps.
The orchestrator never rents or extends compute itself. This app requires a lease
at the train stage even for `--device cpu`, following the F brief's explicit gate.
`cuda|mps|cpu` are accepted; a backend unsupported by the installed trainer reports
BLOCKED. Agent B's current trainer supports CPU and CUDA, not MPS.

The gate records the full sampled scenes (including the focus scene), noise settings,
seeds, fitted physics, and per-scene outcomes for both policies. Both run at 30 Hz
from their declared start_q, with commanded q, the 4-degree divergence reset,
absolute chunks, temporal ensembling, B's command tracker, and the shared success
checker. **PASS** requires a higher mean of clean/real-noise success with no regression
in either condition; ties **HOLD**. HOLD deploys the incumbent. Gate lines name the
incumbent and saved scene-plan hash. Exports are versioned by stage attempt.

`--real` executes C1's guarded runner. Use `--perception-setup <saved setup>`;
otherwise the printed bench command points at `config/perception_setup.json`.
C1's verified map, camera, daemon and perception requirements still apply. This
implementation and its tests have not exercised real hardware, real Gemini calls,
GPU training, or a measured full training/deployment loop.

## Live dashboard sources

Edit **dashboard/config.json** to pin evidence; the server never silently switches
to the newest export or twin. Restart after changing config.

| Setting | Value |
|---|---|
| `real2sim_live` + `real2sim_run` | Parent directory and run name containing `r2s_live/1` run.json |
| `headline_eval` | Pinned export config.json, aggregate eval, or pipeline paired-eval report |
| `game_url` | Actual published Roblox URL; a QR is generated locally with OpenCV |
| `traces` | List of folders containing `r2p_trace/1` episodes |
| `scaling` | Defaults to `runs/scaling.json` |
| `camera_url`, `camera_status_url` | Defaults to C1's publisher on 127.0.0.1:8791 |
| `ttt_status` | Defaults to D's `out/ttt/status.json` |

For a pipeline twin, set `real2sim_live` to
`runs/<task>/r1/twin` and `real2sim_run` to `attempt_001` (or the desired retry).
Pin `headline_eval` to `runs/<task>/r1/eval/challenger.json` for its paired evaluation.

REAL requires explicit real provenance (`source: real`, `mode: real`, or a real/bench
session kind). Synthetic publisher output and attempts are hidden unless
`--show-dry-runs`, which retains a SIM DRY RUN label. `argon_says` appears over the
camera. Missing/offline sources display unknown/empty states, not success numbers.

REAL2SIM computes `gap = 1 - mean(objects[*].iou)` from iteration metrics;
the optimizer's composite score is never treated as IoU. It plots accepted best-so-far
gap and replay motion pixels, marks rejected iterations, and shows best_iter after
completion. SIM/REAL/MOTION badges change at the halfway point of the cross-fade.
Synthetic session provenance stays visible throughout. Motion respects reduced-motion.

GAME counts rounds, success booleans and elapsed trace seconds from metadata/ticks,
separates bots/experts from humans, and shows per-level human coverage and a human
leaderboard. Partial traces are skipped and reread after completion. An unpublished
game has no fabricated URL or QR. POLICY always labels twin/noisy evaluations as
simulation. Scaling accepts `{points: [{episodes, clean, real?}, ...]}` or the array.

The HTTP server uses stdlib only. `/api/state`, `/api/runs`, `/qr.png`, the camera
proxy, and allowlisted image tokens serve the static page. It exposes no arbitrary
file reader or write endpoint. It binds localhost by default; use `--host` explicitly
if the room display connects over the network.
