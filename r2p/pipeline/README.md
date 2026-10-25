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
