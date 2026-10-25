# Agent O: task-derived active experiments

The next-run implementation is isolated from the running bench campaign. No
arm-daemon POST, serial-port access, GPU work, or live Gemini request was made
during this work. Both live-owned files (`r2p/real2sim/active.py` and
`scripts/lead/active_live.py`) are unchanged by agent O.

## New modules

- `r2p/real2sim/experiments.py`: typed task-step proposals, deterministic fallback
  menus, MAIN/high design with refusal feedback, per-skill parameter eligibility,
  and full-trajectory validation. The six shipped task schemas are supported.
- `r2p/real2sim/general_active.py`: physical CPU simulation, independent rendered
  mask observations, bounded identification, whole-experiment hold-outs,
  per-object visual/non-regression gates, inactive candidates and reports.
- `r2p/real2sim/experiment_recording.py`: a shared-budget MAIN scene sensor,
  scene/wrist frame streams with monotonic timestamps, pre-experiment surveys,
  measured-jaw aborts, and multi-landmark sheet checks. Frames survive aborts.
  Contact wrist frames receive a MAIN review for grasp, slip and release.
- `r2p/real2sim/twin_videos.py`: timestamp-aligned CPU command replay into observed,
  sim-before, sim-after, and three-panel MP4s; each output is decoded and checked.
- `scripts/argon/general_active_twin.py` and `scripts/argon/twin_videos.py`:
  separate next-run entry point and existing-run video builder.
- `tests/argon/test_general_experimenter.py`: safety, task relevance, schema,
  refusal feedback, budget client behavior, recording, evidence isolation and
  MP4 tests.

## Task and safety behavior

The menu follows skeleton steps, surveyed objects, shape dimensions and success
predicates. Holds follow the actual source/target path. The sweep joint is chosen
from that path's largest joint change. Bowl tasks include centre and rim-offset
releases above the bowl wall; stacking uses the support height and extra settling;
push tasks approach from a standoff at a sag-safe contact height. Place-on-cell
uses the selected cell and the original supply location. Re-surveying a moved
piece does not move that return destination.

Fits are restricted to the parameters those skills may excite. Static holds fit
compliance; sweeps add damping; contact motions add the relevant object's mass
and friction; pushes add table friction; drops/stacking add contact time constant
and the particular support's materials. Insensitive or rank-deficient parameters
remain explicitly uncertain. Cube and bowl material properties are independent.
Every replay preserves its own surveyed starting positions.

All low approaches/retreats use dense Cartesian interpolation. Every resulting
absolute target is checked for limits, speed, mesh/table clearance, reach,
protected objects and grid cells. The minimum reach is 127 mm. The real path
requires calibrated owner limits and checks measured jaw height in every phase.
The sheet guard compares all labelled landmarks, including rotation, and checks
the initial paper against the saved survey. Physical obstacles are re-surveyed
even when their simulation bodies are fixed. Failure holds through the existing
daemon adapter; no blind retreat is added.

A physical probe exposed a wrist-spin failure in position-only IK. The new
object-aligned orientation constraint keeps roll continuous during a lift. The
regression test bounds lift roll variation. A revised physical MuJoCo probe
successfully placed the cube inside the bowl. A cap probe missed its target cell;
that is reported as a task failure, independently of twin identification.

## Reproduction commands

These campaigns use one CPU process at a time, the real physical `TaskTwin`, and
`FakeGemini` through the normal schema/cache client. Use fresh run names to rerun.

```powershell
.venv/Scripts/python.exe -m pytest -q tests/argon/test_general_experimenter.py
.venv/Scripts/python.exe scripts/argon/general_active_twin.py --task cube_in_bowl --budget 6 --max-nfev 4 --run agent-o-bowl-validated --no-videos
.venv/Scripts/python.exe scripts/argon/general_active_twin.py --task place_on_cell --seed 7 --budget 6 --max-nfev 7 --run agent-o-cell-refined --no-videos
.venv/Scripts/python.exe scripts/argon/twin_videos.py runs/active_twin/agent-o-bowl-validated --fps 3 --width 240
.venv/Scripts/python.exe scripts/argon/twin_videos.py runs/active_twin/agent-o-cell-refined --fps 3 --width 240
.venv/Scripts/python.exe scripts/argon/twin_videos.py runs/active_twin/bench-active-02
```

Each campaign saves `task.json`, `menu.json` (including refusals), two held-out
sessions, four training sessions, `iter_*.json`, `residuals.json`, `status.json`,
`report.md`, `candidate_physics.json`, and `candidate_scene.json`. Videos live in
`videos/<experiment>/`; `videos/manifest.json` records alignment and frame counts.
Apply the physics and scene candidate together if a later evaluation accepts
them. This implementation never activates them.

## Validation and campaign results

Targeted acceptance: **29 passed** (9.25 s). Both final campaigns completed six
experiments: two held-out trajectories with different travel height/speed and
four training experiments. The true hidden stiffness scale was 1.30; both best
candidates recovered it. Both retain the prior contact/material parameters when
the contact refinements fail the gate.

| Task/run | Held-out error before | Best error | Improvement | Jaw RMS before/after (mm) | Object error before/after (mm) | Mean IoU before/after |
|---|---:|---:|---:|---:|---:|---:|
| cube_in_bowl / agent-o-bowl-validated | 1.383275 | 0.435863 | 68.49% | 1.243720 / <0.000001 | 2.285518 / 2.179315 | 0.985169 / 1.000000 |
| place_on_cell / agent-o-cell-refined | 9.949068 | 0.716873 | 92.79% | 1.555183 / 0.436029 | 43.879540 / 2.052675 | 0.943669 / 0.969311 |

The bowl menu executed source and target holds, a shoulder-pan sweep, a centre
drop and a rim-offset drop; it also offered a return probe. All three recorded
pick/drop-to-bowl experiments passed `check_window`. The cell menu used source
and target holds, a shoulder-pan sweep, placement on the selected cell and a
return probe. Both target placements missed the cell. These are experiment and
twin-identification results, not a successful cell policy.

Generated menu details (spec-frame metres; all entries passed validation):

| Task | Source xy | Target xy | Menu |
|---|---|---|---|
| cube_in_bowl | (0.24493, -0.05363) | (0.17190, 0.13031) | source hold, target hold, joint-0 sweep, centre release, release offset +0.02413 m in y, return to source |
| place_on_cell | (0.19961, -0.01416) | (0.18560, 0.10458) | source hold, target hold, joint-0 sweep, target placement, return to source |

Both use 0.09 m travel height, 1 s minimum dwell, 4-degree sweep amplitude and
8-degree/s speed cap. The held-out contact trajectory uses 0.10 m and 7.5-degree/s.
Full trajectories, durations and exact serialized proposals are in each run's
`menu.json`; campaign choices are in `iter_*.json`.

The first cell campaign (`agent-o-cell-validated`, four optimizer evaluations)
retained its prior because all candidates failed a non-regression check. The
final cell run uses the same sampled scene and seven evaluations, without
changing the gate. It accepted the sweep fit. Contact behavior is sensitive to
small servo changes in this scene; the run is not evidence of robust transfer.
Optimizer status, evaluation count and optimality are now retained explicitly.

Broader regression command:

```powershell
.venv/Scripts/python.exe -m pytest -q tests/argon --ignore=tests/argon/test_active_twin.py
```

Result: **176 passed, 1 failed** (66.01 s). The failure is
`tests/argon/test_orchestrator.py::test_offline_end_to_end_under_three_minutes`,
also reproduced alone (45.49 s): the existing `r2p.orchestrator_tools.LocalTools`
imports the legacy `r2p.real2sim.active` path, whose offline active stage reports
`Request failed for gemini-4-argon: Expecting value: line 1 column 1 (char 0)`.
Those existing orchestrator files are owned by another agent and were not edited
here. Logs: `runs/agent_o_argon_regression.log` and
`runs/agent_o_existing_orchestrator_failure.log`. The frozen legacy active-engine
test file and full repository suite were not included in this regression run.

Both final video manifests report **complete**: six sessions and 24 MP4s per
campaign, **48 MP4s total**, with every file decoded and its frame count checked.
Outputs are 3 fps; comparison panels are 720 x 180 pixels. Source footage is
explicitly labelled `SYNTHETIC OBSERVED`. All panels hold together at each camera
capture timestamp, including the final capture; command replay still advances
at 30 Hz. `--continuous-sim` optionally renders extra standalone simulation
frames between camera captures. Representative three-panel captures were also
inspected visually. See `runs/agent_o_bowl_videos.log` and
`runs/agent_o_cell_refined_videos.log` for the full builder output.

The first sampled place-on-cell scene (seed 4) had an occupied placement path;
the Validator refused it. Seed 7 is a task-sampled unoccupied board used for the
reported campaign, with no objects removed after sampling. This is one scene per
task, not a task success-rate or real-transfer evaluation. The synthetic camera
uses seven sparse rendered observations and ideal segmentation. Pose fitting is
independent of hidden simulator poses; failed/occluded contact evidence prevents
identification. Repeated held-out gating is model selection, not a final audit.

## Existing real campaign and next-run handoff

At the final check (2026-10-25, approximately 14:11 Pacific),
`runs/active_live_02.log` had no final result JSON.
`bench-active-02` contained prior scene/physics and raw first-experiment telemetry
(806 sensor rows, 704 commands, 270 capture metadata rows), but no completed
trace/frame session or fitted candidate. The video builder explicitly reports
`waiting_for_campaign` in `runs/active_twin/bench-active-02/videos/manifest.json`.
It does not substitute synthetic footage for missing real recordings.

The separate next-run CLI defaults to simulation. For a supervised future real
run, use `--mover armd --argon live --budget 5 --request-budget 15`, a fresh
`--scene`, and matching calibrated owner limits via `--owner-limits` when the
daemon omits them. The limits JSON contains `joint_map`, `lo`, and `hi`; its map
must match the client. `--plan-only` checks the complete menu without actuator
writes. The default scene sensor and design/review calls share the same hard
15-request client; request/quota/cache failures propagate. The CLI uses the
bench's printed sheet as its guard reference. Other environments can supply
their own labelled-landmark callback to `SurveyedMover`.

Still external: completion of the existing real campaign, its requested videos,
live hardware validation of this next-run path, and any policy/transfer success
claim. The two running files cannot be switched in place until the brief's
required final log result exists.

## Commits

- `3313989`: task-derived engine, recording wrapper, video builder and tests.
- `459b9f6`: stable grasp orientation, shared-budget MAIN sensor/review, calibrated
  owner-limit checks and expanded tests.
- `dafe18e`: frozen supply/sheet anchors, portable camera evidence, meaningful
  held-out motion variants, optimizer diagnostics and timestamp-selection tests.
- `9a8bc98`: capture-aligned comparison videos, final-frame inclusion, optional
  continuous standalone replays, and regression coverage for both cadences.
