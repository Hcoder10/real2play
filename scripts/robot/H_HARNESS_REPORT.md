# Agent H integration handoff — 25 October 2026

The two-round offline harness, software bench path, guarded scripted motion, and
live dashboard are integrated. No serial port was opened, no command was posted
to armd, no GPU was used, and H made no live Gemini request. The phone was read
only through its image GET endpoint. The existing arm daemon was left untouched.

## Run it

From the repository root, using the shared Python environment:

```powershell
.venv/Scripts/python.exe -m r2p run --task place_on_cell --dry-run --rounds 2
.venv/Scripts/python.exe -m r2p run --task cube_in_bowl --dry-run --rounds 2
.venv/Scripts/python.exe -m r2p status
.venv/Scripts/python.exe scripts/argon/tictactoe.py --check
.venv/Scripts/python.exe scripts/argon/tictactoe.py --mock-live --cell r1c1 --publish out/robot/h_mock_live
.venv/Scripts/python.exe scripts/demo_up.py
```

The launcher checks the phone, then serves dashboard `http://127.0.0.1:8800`, run
history `/runs`, read-only camera/status publisher on `8791`, and the authenticated
trace receiver on `8770`. `--check` starts no servers. Set `TRACE_KEY` to the Roblox
server's key, or configure Roblox with the generated `out/demo/trace_key.txt` value.
The key itself is never printed. Occupied ports are not taken over. Ctrl-C closes
only the launcher's servers. Set the published game URL in dashboard/config.json
when available; the dashboard currently displays “Game link pending.”

Repeat either pipeline command to resume. Use `--force r1:session` to rerun all
stages with a fresh probe; existing attempts remain on disk. See
[pipeline details](../../r2p/pipeline/README.md) and [robot notes](README.md).

## Acceptance evidence

Final timings and stage/deployment summaries are in
[`out/h_acceptance/pipelines.json`](../../out/h_acceptance/pipelines.json).
Each task completes 18 stages across two rounds; a subsequent run caches all 18.

| Task | Fresh two-round run | Resume (18 cached stages) |
| --- | ---: | ---: |
| place_on_cell | 91.410 s | 0.218 s |
| cube_in_bowl | 68.029 s | 0.199 s |

Fresh timings use the commands above with `--force r1:session`; resume uses the
commands unchanged. Both returned exit code zero on the final source state.

The actual MuJoCo replay, expert collection, command labeling, tiny CPU training,
paired evaluation, selection gate, and guarded deployment are exercised. The twin
uses recorded synthetic masks and an explicit offline oracle, with zero model calls.

These are smoke tests: two new expert traces per round, five training steps, two
paired scenes and 60 evaluation ticks. Their tied gates retain an evaluated
fixed-hold baseline. Deployment records timeout and `success: false`, with zero
guard clipping and zero commanded-q resets. DONE means the stages executed; it
does not qualify the resulting policy for the physical demo.

All nine measured-cell plans pass IK, swept clearance and joint bounds. With the
25 mm wide, 14 mm tall piece, the minimum checked finger clearance is 10.7 mm.
The current supply is loaded from the task (x=0.15, y=-0.092 m), rather than the
older supply coordinates in the brief. Full targets, owner units and geometry:
[`out/ttt/check.json`](../../out/ttt/check.json).
The updated 32×16 mm bottle-cap geometry also passes all nine no-motion plans,
with 12.6 mm minimum checked finger clearance; see
[`out/ttt/check_current_caps.json`](../../out/ttt/check_current_caps.json).

All nine offline pad-contact placements reach the goal: clipping is 1.27–1.44%,
with zero refusals and zero commanded-q resets. Evidence:
[`out/h_acceptance/cells.json`](../../out/h_acceptance/cells.json). The center-cell
MockRobot test with read-only phone viewing also succeeds (1.33% clipped ticks,
zero resets, approximately 13.2 seconds wall time). This does not demonstrate a
physical placement or perception-controlled placement.

Real-time dashboard screenshots at 1920×1080:

- [Live](../../dashboard/shots/live-1920x1080.png), with [DOM/error evidence](../../dashboard/shots/live-1920x1080.json).
- [Mock](../../dashboard/shots/mock-h-1920x1080.png), with [DOM/error evidence](../../dashboard/shots/mock-h-1920x1080.json).

Both have no overflow or JavaScript errors; the visible image badge matches the
dominant image, and the tic-tac-toe model/source label remains visible. Live mode
loads G's recorded twin, the actual phone feed, real trace folders, exports, run
manifests and the tic-tac-toe status file. Missing measurements remain missing.

## Fixes and files

- `r2p/robot/`: pace absolute chunks within existing speed caps, honor requested
  speed, retain distinct pitch-joint sag gains, prevent double compensation for
  servo-target exports, advance learned-policy commands from observed state, and
  record commanded q. Synthetic grasps require closing pad contact, remain rigid
  in the gripper frame, and release when the pads separate. Safety limits and the
  four-degree divergence threshold were not loosened.
- `scripts/argon/tictactoe.py`, `scripts/robot/`: measured task supply, short-piece
  jaw targets, safe near-row approach/retreat, cylinder support, no-motion check,
  MockRobot/live-phone mode and guarded direct RealRobot setup after daemon release.
- `r2p/pipeline/`, `r2p/cli.py`, `r2p/policy/train.py`: bounded CPU dry defaults,
  offline twin fixture, optional-object and board-marker collection fixes, retry
  handling, explicit smoke metadata, honest deployment results and resume support.
- `dashboard/`, `scripts/demo_up.py`: evidence discovery, real/synthetic provenance,
  read-only publication, tic-tac-toe wiring, real-time screenshots and one launcher.
- `tests/app/`, `tests/robot/`, `tests/argon/test_integration.py`, `tests/sim/__init__.py`:
  integration regressions, active-map expectations, publisher provenance and full
  suite collection compatibility.

First H commit: `827b6f2` (`fix(robot): pace safe probes and harden measured
tic-tac-toe bench path`), preceded by 851 passing tests. The final integration
commit includes this report and the remaining pipeline/dashboard fixes; its hash
is in the handoff message.

The required complete suite passed **862 tests in 143.70 seconds** before the
final integration commit:

```powershell
.venv/Scripts/python.exe -m pytest -q tests/core tests/argon tests/perception tests/robot tests/app tests/sim tests/policy
```

## Remaining demo dependencies

The active joint map is still `verified: false`. Direct real mode requires the
lead to rest the arm and stop armd, then explicitly use `--real --arm-released`.
An active or uncertain daemon ownership check refuses serial access. An unverified
map additionally requires `--allow-unverified`, which halves motion speed. H did
not run this mode or validate physical calibration, placement, camera tracking,
learned-policy performance, or real-time inference on the hardware.

The actual bottle caps are now specified as about 32×16 mm; the requested small-
piece check retains its explicit 25×14 mm default. Pass measured `--piece-size`,
`--piece-height` and `--supply-xyz` overrides for the object used at the bench.
Full-size policy training/evaluation belongs to the lead's leased GPU run. The
published Roblox URL, matching upload key, human play data and production policy
selection remain external demo dependencies. No real success rate is inferred
from the offline fixture or illustrative dashboard mode.
