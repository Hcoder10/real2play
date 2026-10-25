# Agent G live Argon QA — 2026-10-25

59 of 80 authorized wire requests used. All calls are logged in
[`runs/agent_g/calls.jsonl`](../../runs/agent_g/calls.jsonl), with a persistent pre-send budget in
[`request_budget.json`](../../runs/agent_g/request_budget.json). No serial connection, daemon POST,
hardware motion, GPU job, fallback to SMALL for MAIN reasoning, or active calibration change occurred.
The [evidence manifest](../../runs/agent_g/evidence_manifest.json) hashes the 40 audit inputs/results;
`calls.jsonl` SHA-256 is `de594b8451b133b1ef4046d8a05117f2853cd1bc1739aff9539b23a9bc05b5b0`.

## Measured results

Latency below is p50/p90 **per wire attempt**, including retries and unsuccessful early versions;
cache hits are excluded. It is not whole-loop wall time. The JSON summary's `errors` counts terminal
transport errors; the retried 504 is visible in `attempt_details` even though its final call succeeded.

| Path | Wire requests | p50 / p90 seconds | Accuracy / outcome |
| --- | ---: | ---: | --- |
| Tic-tac-toe | 21 MAIN | 4.984 / 5.922 | Real empty frame correct; 20/20 composite boards exact, 180/180 cells correct, 20/20 legal decisions |
| Visual fingertip pointing | 14 MAIN | 4.281 / 5.674 | All 14 real pose photos processed; fitted candidate fails the activation gate |
| Twin detection and critique | 15 (3 SMALL, 12 MAIN) | 6.515 / 9.766 | Final CLI: cube IoU 0.841, centroid 2.985 px; bowl IoU 0.952, centroid 0.786 px, four held-out frames |
| Task authoring | 4 MAIN | 4.586 / 6.731 | Three early schema failures; corrected authoring succeeds in one 7.500 s request and validates |
| Perception boxes | 2 SMALL | 2.961 / 3.029 | Cube fit 5.638 px and seen; partly clipped bowl 9.660 px and correctly marked weak/stale |
| Label refinement | 3 MAIN | 15.531 / 39.293 | One real 45.234 s HTTP 504 retried; corrected typed schema returns a complete valid command partition |

Board end-to-end latency was **5.094 / 6.062 s**. Cache-only replay repeated all 20 boards with
100% exact/legal results, zero requests, and **0.062 / 0.063 s** end-to-end latency.
Summaries: [live](../../runs/agent_g/boards_summary.json),
[cache replay](../../runs/agent_g/replay/boards_summary.json),
[all wire attempts](../../runs/agent_g/summary_summary.json).

The final twin keeps iteration 0 and rejects a substantially worse proposed iteration 1 before
the critic stops. `run.json`, scenes and 1280x480 overlays are atomic `r2s_live/1` dashboard artifacts:
[live CLI run](../../out/live/agent_g_cli_final/run.json),
[current-code cache-only verification](../../out/live/agent_g_cli_verified/run.json).
The latter reproduced exactly the same metrics with zero network requests.
Task authoring's own final twin reached cube/bowl IoU **0.816 / 0.866** with centroid errors
**3.731 / 3.764 px**: [task run](../../out/live/agent_g_task_v3/run.json).
These are agreement measurements against independently tracked image masks, not surveyed metric truth.

## Evidence boundaries

- The background and cube crop are real bench photographs; the empty-board check is a fresh read-only
  phone-camera frame. Twenty occupied boards are **photo composites**, including one coin in every
  cell and ten mixed legal states. They establish image-read accuracy, not physical gameplay accuracy.
  [Manifest](../../runs/agent_g/boards/manifest.json), [contact sheet](../../runs/agent_g/boards/contact.jpg).
- Penny and nickel photographs came from Wikimedia Commons because no local bench coin photograph
  was available: [US One Cent Obv](https://commons.wikimedia.org/wiki/File:US_One_Cent_Obv.png) and
  [Nickel obverse](https://commons.wikimedia.org/wiki/File:Nickel_obverse.png).
  The 25 mm cube sprite was cropped from `runs/bench/flat2.jpg`; actual occupied boards were not moved.
- The twin session contains the 14 real `runs/bench/vcal/p*.jpg` frames and measured owner-degree poses.
  Its `r2p_trace/1` has **no command ticks**. Measured MJCF pose samples and presentation-only frame times
  are explicitly metadata, with `training_eligible=false`. Static replay reports null trajectory error;
  it supplies no servo, sag, friction, timing, grasp, release, or task-success certification.
- Label refinement used the available recorded **mock/bot** 1362-tick, 45.4-second placement trace at
  `runs/place_on_cell-dry/r1/session/attempts/generation_001/20261025_115448_491190/attempt_01/trace.jsonl`.
  There was no real motion trace to label. Its valid result is
  `[0,23.6,pick]`, `[23.6,32.866667,place_at]`, `[32.866667,42.7,release]`, `[42.7,45.4,home]`;
  exact command strings and boundaries are in [label_summary.json](../../runs/agent_g/label_summary.json).
- The shared task was subsequently changed by its owner to white bottle caps. Live board accuracy above
  applies to the requested brown pieces and coins. It does not certify bottle-cap classification.

## Visual calibration result

The product path is `r2p/sysid/visual.py` and `scripts/argon/visual_calib.py`:
plan -> injectable collect -> resumable Argon point -> soft-L1 fit -> independent leave-one-out report.
Synthetic tests cover swapped endpoints, offset recovery, signs, outliers, stale frames and resume.
Collection has no implicit robot transport; the lead supplies the guarded mover and fresh-frame adapter.

With 14 fresh MAIN observations, initial tip RMS was **11.377 px**, training RMS **5.262 px**,
LOO RMS **5.652 px**, median **4.161 px**, and p90 **8.725 px**, rank 5.
The default p90 <=5 px gate **fails**. The candidate remains unverified and inactive; no millimetre
accuracy claim is made. [Fit JSON](../../runs/agent_g/visual/fit.json),
[report](../../runs/agent_g/visual/report.md), [pointing evidence](../../runs/agent_g/visual/pointed.json).

## Bugs fixed

1. Central Gemini transport now has a finite 45 s timeout, disables hidden SDK retries, bounds explicit
   retries with 1/2/4 s backoff, counts retries/repairs against request budgets, and logs each attempt's
   latency, status and actual answering model. Daily quota failures remain explicit. Fake tests cover
   503, timeouts, RPM/daily quota, repair, fallback and cache-only behavior; a live 504 verified retry.
2. Board projection uses the measured rotated cell grid, resized camera intrinsics and distortion-aware
   remapping. Labels no longer cover piece centers. Coin/cube instructions reject gripper, shadow and
   off-board distractions. An empty board correctly waits for the human instead of ending the live game.
3. An empty nested task JSON schema produced `{task:{}}` on all three early live calls. Explicit nested
   task/goal/skeleton schemas fix decoding. Generated poses are marked inferred; calibrated fitted poses
   and locally generated randomization are published together, and perception cannot treat a guessed
   fixed bowl position as surveyed ground truth.
4. SMALL detection is single-frame; shape dimensions are validated (bowl/cylinder use diameter,height).
   Independent reference-based masks avoid background and same-color object leakage. Clipped projections
   fit visible sensor silhouettes. Rejected proposals cannot tighten incumbent keep thresholds. Network,
   quota, JSON and budget stops preserve the best fitted scene.
5. Static calibration photos render the measured arm poses without fabricated control targets or dynamics.
   Real camera distortion/extrinsics are retained and panels are letterboxed rather than stretched.
6. Labeling now requests typed integer tick intervals with an enumerated command vocabulary. It validates
   the complete partition and permits one semantic correction before returning seconds. The previous
   heterogeneous tuple schema caused command strings containing malformed numeric/JSON fragments.
7. Randomized supply coordinates no longer masquerade as known measurements; frozen or explicitly
   verified supply positions remain supported. Color thresholds preserve shaded foreground sides of
   white pieces while component tracking rejects distant same-color distractors.

## Reproduction

Run from the repository root. Live stages share one root and run sequentially. Do not rerun collection
against the bench without its owner supplying the adapter. Fresh twin outputs require a new directory.

```powershell
.venv/Scripts/python.exe scripts/argon/live_qa.py summary --root runs/agent_g
.venv/Scripts/python.exe scripts/argon/live_qa.py prepare --root runs/new_qa
.venv/Scripts/python.exe scripts/argon/live_qa.py empty --root runs/new_qa
.venv/Scripts/python.exe scripts/argon/compose_boards.py --frame runs/agent_g/live.jpg --penny runs/agent_g/assets/penny.png --nickel runs/agent_g/assets/nickel.png --cube runs/agent_g/assets/cube.png --cube-base 37 96 --out runs/new_boards
.venv/Scripts/python.exe scripts/argon/live_qa.py boards --root runs/new_qa --manifest runs/new_boards/manifest.json
.venv/Scripts/python.exe scripts/argon/live_qa.py task --root runs/new_qa --out-live out/live/new_task
.venv/Scripts/python.exe scripts/argon/live_qa.py visual --root runs/new_qa
.venv/Scripts/python.exe scripts/argon/visual_calib.py plan --out runs/new_visual_plan
```

For replay, set `$env:R2P_CACHE_ONLY = '1'` before `boards` or the exact twin CLI below, then
`Remove-Item Env:R2P_CACHE_ONLY` when finished. Changed task or image inputs deliberately cause cache misses.

```powershell
.venv/Scripts/python.exe scripts/argon/run_twin.py runs/agent_g/session --task cube_in_bowl --out out/live/new_cached_twin --iters 2 --max-calls 4
.venv/Scripts/python.exe -m pytest -q tests/core tests/argon tests/perception tests/robot tests/app tests/sim tests/policy
```

## Validation and commits

First implementation commit: `f80c069` — `Harden Gemini transport and board reading; add visual calibration pipeline`.
This report accompanies the second increment, which contains real-input fixes, the audited QA scripts,
regression coverage and documentation. Every staged path belongs to Agent G; other owners' edits are retained.

Final acceptance: **861 passed in 177.29 seconds**, using the exact complete-suite command above.
Log: [full_suite_complete.txt](../../runs/agent_g/full_suite_complete.txt). This includes the perception
regressions exposed by the owner's concurrent bottle-cap task change (`65f8425`); the simulator expert
acceptance also passes with that final task. The current-code twin CLI and all 20 board cases were
separately replayed cache-only. `git diff --check` passes on Agent G's paths.

The shared tic-tac-toe CLI also contains its owner's pending bottle-cap placement/description edits.
Agent G stages only the human-turn wait and coin-description changes in that file; the full-suite
result describes the shared working tree. The owner's remaining edits are preserved.

Remaining evidence: obtain a better separated/visible tip collection to pass calibration, real command
traces for physics identification, physical multi-piece board checks (including the newly selected caps),
and a view containing the whole bowl. No additional live requests are needed to review these results.
