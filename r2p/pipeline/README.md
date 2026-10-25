# Offline harness and real evidence

From the repository root:

```
.venv/Scripts/python.exe -m r2p run --task place_on_cell --dry-run --rounds 2
.venv/Scripts/python.exe -m r2p run --task cube_in_bowl --dry-run --rounds 2
.venv/Scripts/python.exe -m r2p status
```

Dry mode defaults to two expert episodes per round, five training steps on CPU,
a 32-wide model, and two paired scenes with a 60-tick evaluation budget. Model
exports also record all three noise conditions (one scene each, 60 ticks).
The synthetic probe runs through MockRobot and the guarded 30 Hz runner; recorded
images are 2 Hz with explicit tick timestamps and camera calibration. The actual
MuJoCo replay/metrics, expert collection, deterministic command segmentation,
training, paired gate, and guarded deployment all run. The synthetic twin is
initialized from the fixture's scene/masks with an explicitly named offline oracle
critic. This makes zero Gemini requests and does not claim an Argon-fitted real twin.

When there is no incumbent, dry mode creates a named, evaluated fixed-hold baseline.
It is deliberately a weak baseline. A tie is HOLD, and its unsuccessful four-second
deployment is recorded honestly. DONE means every stage finished, not that a tiny
policy succeeded at the task. The manifest profile, export metadata, dashboard and
deployment summary preserve those distinctions. These runs do not qualify a policy
for the live demo. Use full-size training/evaluation and supervised hardware evidence
for that decision.

Repeat a command to resume: completed stages with unchanged inputs and valid hashed
artifacts are CACHED. Extend `--rounds` without recomputing earlier rounds. Use
`--force r1:session` for a fresh probe and all descendants, or `--force r2:train`
to replace only that training stage and its descendants. Attempts use distinct
generation directories. Corrupt or crashed recordings cannot count as completed
attempts; a successful retry can supersede them without deleting the old evidence.

`--traces`, `--steps`, `--eval-n`, and `--incumbent` override defaults. Dry evaluation
remains intentionally short. Real/plan modes retain the full defaults and require an
existing incumbent. CPU training needs no GPU lease; accelerator workers still
validate the landlord lease and apply the leased VRAM cap before training.

The command-only acceptance suite is:

```
.venv/Scripts/python.exe -m pytest -q tests/core tests/argon tests/perception tests/robot tests/app tests/sim tests/policy
```

## Game data through the reference twin

Use the receiver's completed JSONL directory (port 8770, `X-Trace-Key` upload):

```powershell
.venv/Scripts/python.exe -m r2p loop --from-game roblox/out/traces --task place_on_cell
```

An offline collection and training run needs no Studio, hardware, GPU or Gemini:

```powershell
.venv/Scripts/python.exe -m r2p loop --from-game roblox/out/my_headless/traces --collect-headless 8 --task place_on_cell --output runs/my_game_loop --steps 1000 --eval-n 12
```

Use a fresh output directory. `--collect-headless` runs a Python port of the
actual game's `SkillOperator` and `Kinematics.ik` against MuJoCo contacts. Its
traces declare `source=bot`, `headless=true`, and describe their simulated origin.
They are neither human play nor evidence of Roblox-engine contact parity.
The collector supports pick/place_at/release/home/wait; replay accepts all tasks.
The port can be checked against the actual pure-Luau controller on recorded inputs:

```powershell
.venv/Scripts/python.exe -m r2p.pipeline.game_parity roblox/out/my_headless/traces/headless_00003026.jsonl --out out/player_parity.json
```

The loop validates through `r2p.traces`, requires the selected task revision,
rejects test fixtures and duplicates, and replays every target from the initial
scene through a frozen snapshot of `config/physics/so101_bench.json`. It never
teleports an object to match a recorded pose. Default acceptance limits are
1 degree joint RMS, 4 degrees maximum joint error, 2 mm jaw RMS, 5 mm maximum
object-position error and 5 degrees symmetry-aware yaw / object-axis tilt error. Mismatch
means "not reproducible in this reference twin", not proof a player cheated.
Old physics fingerprints are recorded; acceptance depends on current replay.

Accepted rows use the twin's observations, previous commanded q with the
4-degree reset rule, and original absolute servo targets. `r2p.labeling.segment`
attaches the Argon command vocabulary from game phase events, with **zero model
requests**; these labels are not claimed to be Argon-refined. Training constructs
15x6 absolute chunks, padded at command boundaries, and mixes all accepted
successes/failures with experts (the existing dataset downweights failures).

Two equal-size CPU networks train for equal updates/seeds: experts only, then
experts plus accepted game traces. The saved paired plan uses distinct training
and evaluation scene hashes, 12 full held-out scenes by default, clean and
real-noise conditions, and the current physics. `--max-ticks` explicitly opts
into truncated smoke evaluation. A tie or regression yields HOLD. There is no
hardware deployment stage and no automatic claim that more data improved success.
The offline high-level scheduler advances from lowered placement to the separate
release command without requiring the placing policy to have already opened the
jaws; final success still uses the unchanged shared checker and settled window.
This scheduler makes no model requests and is recorded in the evaluation plan.

Optionally add `--validation-game <other-dir>` for held-out replay action-error
diagnostics. These traces are replayed/filtered but never trained on; overlapping
training scenes fail validation. `imitation_eval.json` reports per-episode and
aggregate absolute-chunk MAE in radians. Prediction error is not a task-success
rate and never changes the selection gate.

The run retains raw inputs, replay metrics/rejections, labeled traces, expert
traces, both self-contained exports, scenes, per-scene outcomes, hashes and gate.
After completion an atomic `runs/game_loop_latest.json` publishes GAME source/
rejection counts and the POLICY before/after curve. The dashboard respects an
explicit `headline_eval` pin. Simulated bots remain separate from human counts.
Failed runs retain their manifest and do not replace the last publication.

For real human data, publish the Roblox place, configure its server-only HTTPS
receiver URL and matching key, and collect complete rounds. No publish or live
Gemini request is part of this command. Changed task revisions need recollection
or an explicit `--task <recorded-task.json>`; never silently mix incompatible goals.
