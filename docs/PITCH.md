# Real2Play: pitch and live demo

Gemini 4 Argon hackathon, SHACK15 SF, Sun Oct 25 2026. Problem Statement Two, "More Dimensions":
render what you made, watch it, critique it against the goal, fix it in a loop. **Our goal is reality itself:
the real robot's camera.**

Rule for whoever is speaking: quote only the numbers in this file (all measured today unless marked
simulation). If a number changes at the bench, change it here first. Architecture page:
[docs/architecture.html](architecture.html) (open it full screen on the projector).

---

## 1. The 90-second pitch (spoken, ~230 words)

> **[Stand next to the arm. Dashboard on the projector.]**
>
> "Our user is a three-person robotics startup with one SO-101 arm and no data budget. That was us this
> morning. To teach this arm a task you need thousands of demonstrations, and every one costs a person
> sitting at the robot. Simulators are free, but they are wrong in ways you only find on the real robot.
>
> Real2Play makes reality the goal of Gemini 4 Argon's render-watch-critique-fix loop. Argon moves the
> real arm, watches it through a calibrated phone, points at the jaw tips, and fits the arm's joint offsets:
> reprojection error from 84 pixels down to 7. It also caught an error we missed: the wrist roll was
> about 190 degrees off, invisible until it asked to see the jaw open.
>
> Then it renders its own twin over the real frames, critiques the mismatch and patches it, keeping only
> changes that improve held-out frames: object overlap 0.84 and 0.95. From that twin, Argon running as an
> Antigravity managed agent builds a Roblox game: the same success checker, 442 shared test vectors,
> 4.97 millimetre parity with the twin. People play; every round replays through the twin and becomes
> training data, labelled with Argon's commands.
>
> At run time Argon is the brain: it reads the board, 20 of 20 correct, about 5 seconds a turn, and says
> pick and place. A learned policy is the hands, at 30 hertz.
>
> Let's play."

Why it needs Argon (if asked, or as a 10-second insert after "Simulators are free"): *"It needs a model that can
watch long video of a real robot, point at millimetre features, write a multi-file game and test it, and
judge its own render against reality, in one loop with no human in the middle. That is Argon."*

---

## 2. What we measured today (the only numbers to quote)

| Link | Number | Status |
|---|---|---|
| Scene camera (Pixel phone) from an ArUco board | 0.70 px reprojection; the board flush against the robot base is the ruler (no tape measure) | real bench |
| Printed grid, measured by the camera | 36.3 mm cells (printed at ~73 % scale) | real bench |
| Argon visual calibration of the arm | 14 poses, Argon points at jaw tips, robust fit of joint offsets: 84 px -> 7 px train; leave-one-out median 8.8 px (~5 mm) | real bench |
| Hidden calibration error Argon caught | wrist roll unobservable with the jaw closed; 3 photos with the jaw opening wider, Argon separates fixed finger vs moving jaw: wrist roll ~190 deg off, gripper opens much wider than the model; 130 px -> 6.5 px | real bench |
| Real pick-and-place | bottle cap onto the centre cell, 14 mm off; a second autonomous attempt (Argon located the supply caps) picked and carried the cap | real bench, scripted IK (not the learned policy) |
| Argon tic-tac-toe brain | 20/20 boards read correctly, all moves legal; latency p50 5.1 s / p90 6.1 s | real frames with composited boards |
| Twin visual match | object IoU 0.84 (cube) / 0.95 (bowl) on held-out real frames | real frames |
| Servo model sag | emerges in the twin: 3.3 mm at 18 cm, 9.8 mm at 30 cm (real arm: 9-12 mm at 30 cm) | simulation vs real |
| Expert in the twin | 87.5 % (224/256), whole board pose and cap size randomized | simulation |
| Policy | 3-seed ensemble, absolute 15x6 joint-target chunks at 30 Hz, trained on thousands of expert traces on the RTX PRO 6000 rig | simulation; **not yet run on the real arm** |
| Roblox harness | lobby of ~20 tables, sit to play, preset tools in configurable levels; success checker ported to Luau, 442 shared vectors; twin parity 4.97 mm | built; **game not yet published** |
| Pipeline | `python -m r2p run --task place_on_cell`: probe -> twin -> data -> label -> train -> eval -> gate -> deploy, 18 stages, resumable; 860+ tests | software |

Say it straight when asked: the learned policy has not run on the real arm yet; the Roblox game is not yet
public; active-twin experiments on the real arm are in progress. What runs live on the arm is Argon's brain plus
our calibrated IK.

---

## 3. Three-minute live demo: run-of-show

**Setup before the slot (T-10 min), Windows laptop, repo root, `.venv/Scripts/python.exe` as `py`:**

1. Arm powered, armd running and at rest (owner present for anything that moves).
2. Phone camera server up on `127.0.0.1:8792` (`scripts/lead/phonecam.py`). Check: `curl 127.0.0.1:8792/list`
   shows `age_s` under 1.
3. `py scripts/demo_up.py` -> dashboard `http://127.0.0.1:8800` on the projector; `docs/architecture.html` in a
   second tab.
4. Board taped at its measured pose, 4-5 white bottle caps on the supply line at the robot's right, coins for the
   judge. Grid cells empty.
5. One warm-up Argon call so the first live call is not cold: `py scripts/lead/ttt_armd.py --find-caps`
   (prints the supply caps; no motion).
6. Backup evidence open in an image viewer, one Alt-Tab away: `runs/agent_g/boards/contact.jpg` (the 20 boards
   Argon read) and the recorded real placement in `runs/bench/video/`.

| Time | Who / where | Exact step | What the audience sees | Fallback |
|---|---|---|---|---|
| 0:00-0:20 | Speaker at the arm | Opening lines of the pitch (problem + named user). | Real arm at rest, dashboard REAL panel with the live phone view. | Phone view frozen: say "that's the phone camera", switch the projector to `architecture.html`; do not debug on stage. |
| 0:20-0:50 | Speaker at projector | `architecture.html`: click **Real** then **Calibration loop**. Point at 84 -> 7 px and the wrist-roll catch. | The chain Real -> Twin -> Game -> Play -> Replay -> Policy -> Real, loops highlighted, measured numbers on click. | Page will not load: it is a single offline file; open it from disk (`docs/architecture.html`). |
| 0:50-1:00 | Judge at the board | Judge places a coin (X) on any cell. Operator runs `py scripts/lead/ttt_armd.py` and presses Enter. | The judge's coin on the paper grid. | Script not started: start it now; the first prompt is "Place your coin (X), then Enter". |
| 1:00-1:10 | Argon (live) | Argon reads the phone frame, returns board + move + "why". Speaker reads the "why" aloud from the terminal / dashboard tic-tac-toe panel. | Board state and Argon's reasoning; `model: gemini-4-argon`, latency ~5 s. | **Argon slow** (no reply by ~12 s, p90 is 6.1 s): keep talking through the twin node on `architecture.html`; the transport times out and the code falls back to local minimax on the last confirmed board, labelled `source: minimax`. Say so. **Argon wrong move**: every Argon move is checked against minimax; a worse or illegal move is rejected and retried once, then minimax plays. **Board unreadable**: the script asks the operator to confirm the human move; the operator reruns with `--place <cell>`. |
| 1:10-1:40 | Arm (live) | Argon finds a supply cap (`ttt:caps`); armd checks every path; the arm picks the cap, carries it, places it on Argon's cell, parks. | The real robot plays its move. | **Arm refuses** (armd clearance/limit check, or path refused): never force it. Say "the safety layer refused that path, which is the point", then `py scripts/lead/ttt_armd.py --park`, and run one known-good placement: `py scripts/lead/ttt_armd.py --place r1c1 --cap 0` (fixed supply position, no Argon). **Grasp misses** (gripper closes on nothing; the result prints `grasped: false`): hand the cap to the cell yourself, say "missed by a few millimetres, the 14 mm number is why we're building the twin", continue. **Arm faults / no torque**: stop motion, show the recorded real placement from `runs/bench/video/` and keep going. |
| 1:40-2:00 | Judge + Argon | Judge plays a second coin, Enter; Argon reads and moves again (optional: skip if behind time). | A second Argon turn, consistent board history. | Behind time: press `q` to stop the game, go straight to 2:00. |
| 2:00-2:30 | Speaker at projector | Dashboard REAL2SIM panel: real frame vs twin overlay, IoU 0.84 / 0.95, rejected iterations visible. Then GAME panel: tables, levels, 442 vectors, 4.97 mm parity. | The loop: render, watch, critique, fix, with rejected rounds shown honestly. | Dashboard panel empty (source offline): use the Twin and Game nodes on `architecture.html`, same numbers. |
| 2:30-2:50 | Speaker | POLICY: "Argon says pick(piece), place_at(piece, cell); the policy does 30 Hz control. Expert 87.5 % in the twin; the learned policy is trained on the rig and is next on this arm." | `architecture.html` Policy node. | None needed (static). |
| 2:50-3:00 | Speaker at the arm | Close: "Show Argon your robot. It makes the twin exact against reality, turns your task into a game the world plays, and trains your robot on that play." | The arm, parked. | If over time: drop this sentence; the demo already closed the loop. |

**Hard rules on stage.** Owner within reach of the arm for every motion. No `--allow-unverified` live. Never
kill or restart armd during the slot; if it is wedged, the scripted fallback is the recorded video. Do not
run the learned policy on stage unless it was run on this arm before 17:00 and the result is written in
section 2.

**Camera stall, in detail.** `ttt_armd.py` waits for a frame fresher than the request, so a frozen phone
blocks the turn instead of reading an old board. Signs: dashboard REAL panel age climbing, script silent after
Enter. Fix in under 20 s: unlock the phone / bring the camera app to the front; if the server is down, restart
`scripts/lead/phonecam.py` in its window (the camera pose is calibrated and the phone must not move). If it is
not back by 20 s: Ctrl-C, narrate the turn from `runs/agent_g/boards/contact.jpg` (20/20 read), and run the
fixed placement `--place <cell> --cap 0`, which needs no camera.

---

## 4. Likely questions

- **Why not let Argon drive the arm directly?** About 5 s per call (p50 5.1, p90 6.1 today). Control needs 30 Hz.
  Argon decides what to do, the policy decides how to move.
- **Sim-to-real gap?** That is what the loop is for: Argon renders the twin over real frames and patches it until
  held-out frames agree; the twin's servo model reproduces the real arm's sag (9.8 mm vs 9-12 mm at 30 cm).
  The policy is state-based, so the game graphics never have to look like the camera.
- **Is crowd data any good?** Every episode is checked server-side by the same success checker as the twin (442
  shared vectors), replayed through the twin, and physically impossible ones are rejected before training.
- **What is not done?** The learned policy has not run on the real arm; the Roblox game is not published; active
  experiments on the real arm to identify physics are in progress. Everything above the line in section 2 marked
  "real bench" was done on this table today.
- **Keeping people in control?** armd checks every path against limits and clearance and refuses unsafe ones; a new
  policy deploys only if it beats the incumbent on identical scenes; neither the game builder nor the task
  author publishes anything on its own: their output must pass local tests and the owner publishes.

---

## How Gemini 4 Argon is used

Every reasoning and perception call goes to `gemini-4-argon` through the Interactions API (`r2p/gemini.py`) with
an explicit `thinking_level`. `gemini-3.8-flash` is off: it needs an explicit owner approval flag, and board,
critique and experiment reasoning are never allowed to fall back. Every call is cached by input hash and logged
(`calls.jsonl`: served model, latency, attempts), so the whole demo can be replayed with zero requests.

| Job | Call site | What Argon does | Thinking | Today |
|---|---|---|---|---|
| Board reading and move choice | `ttt:decide:*` (`r2p/ttt`) | Reads the real phone frame (projected cells + top-down view), returns the board, a legal move and a one-line reason; locally validated against history and minimax | medium | 20/20 boards, all legal, p50 5.1 s / p90 6.1 s |
| Cap finding | `ttt:caps` (`scripts/lead/ttt_armd.py`) | Finds bottle-cap rim centres in the supply zone and marks caps already on the grid; deprojected at rim height through the calibrated camera | low | second real attempt picked and carried an Argon-located cap |
| Jaw-tip pointing for calibration | `visual:point` (`r2p/sysid/visual.py`), `vcal_tips` (`scripts/lead/visual_calib.py`) | Points at the fixed finger and moving jaw tips in photos of known commanded poses; a robust fit solves joint offsets | low | 84 -> 7 px, LOO median 8.8 px (~5 mm); wrist roll ~190 deg off found, 130 -> 6.5 px |
| Scene extraction and initial boxes | `real2sim:detect`, `perception_boxes` | Finds task objects and their sizes in real frames to seed the twin and perception | low | feeds the twin below |
| Twin critique and patching | `real2sim:critique:*` (`r2p/real2sim/loop.py`) | Compares real frames with twin renders of the same recorded state, proposes a bounded scene patch; local held-out keep rule accepts or rejects it | medium | object IoU 0.84 / 0.95 on held-out frames; a worse iteration was rejected |
| Experiment design | `active_choose`, `active_patch` (`r2p/real2sim/active.py`) | Picks the next safe experiment (hold, sweep, push, pick) that most reduces twin uncertainty, then proposes physics patches; held-out experiments decide what is kept | high | simulation campaign recovered hidden stiffness/friction/mass (held-out error -95 %); real-arm campaign in progress |
| Task from video | `task_from_video:*` (`r2p/real2sim/task_video.py`) | Watches a whole recorded attempt and writes the task spec (objects, goal predicates, skill skeleton) | low | used when the pipeline gets a task in words plus a recorded attempt (`--task "..." --video <dir>`) |
| Trace labelling | `labeling:refine` (`r2p/labeling.py`) | Segments each play or expert episode into Argon commands: pick(piece), place_at(piece, cell), release, home | low | complete valid partition on a recorded trace |
| Task authoring | `author_task` (Antigravity managed agent, `scripts/argon/author_task.py`) | Argon in a Google-hosted sandbox writes a new task and positive/negative checker vectors; must pass our checker and scene sampler locally | agent-managed (high effort in prompt) | dry run passes the real checker and scene sampler; live managed run gated the same way |
| Game building | `build_game` (Antigravity managed agent, `scripts/argon/build_game.py`) | Argon in a Google-hosted sandbox receives the task, measured board, physics, camera and harness sources, writes the Roblox game configuration and runs sandbox tests; applied locally only if all Luau suites and trajectory parity pass; never publishes | agent-managed (high effort in prompt) | dry run: all five Luau suites pass, joint RMS 0.009 deg vs twin; live managed run in progress |
| JSON repair | `*:json_repair` (`r2p/gemini.py`) | One bounded schema repair of a malformed reply | low | at most one repair per call, counted against the request budget |
