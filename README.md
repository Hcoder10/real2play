# Real2Play — an AI that turns a real robot into a game, then into a robot policy

> An autonomous Gemini 4 Argon system that takes a **real robot bench**, **experiments on the real arm one move at a time** until its **own simulation matches reality**, **builds a multiplayer Roblox game from that twin** (via an Antigravity managed agent), **collects play data**, **replays it back through the twin**, **trains a policy on a rented B200**, and **deploys it to the real arm with Argon giving the commands**, with **no human demonstration**.

Built for the **Gemini 4 Argon** hackathon (Problem Statement Two: *More Dimensions* — render what it made, watch the result, critique it against the goal, fix it in a loop). Here the goal is **reality itself**.

---

Hey judges!

This is a large project, and a lot of my friends that I demoed it to got bored by the 10th minute of explaining, asking me to "just show it to them working". I want to fix that for you guys.

# I propose:

A 7 minute plan for you to fully experience the project while seeing it in action.

I only ask that you watch/read/experience what i've assembled, and maybe vote for me to win :)

# Without further ado, this is Real2Play:

I want to start by explaining WHY you should consider this to win:

1. Gemini 4 Argon **experimented on a real SO-101 arm by itself** — one move per turn, looking at a calibrated phone camera *and* the wrist camera after every move — and **picked a bottle cap and placed it in the centre tic-tac-toe cell in 23 turns**, then **grasped a brown box and dropped it into a blue bowl in 28 turns**, with zero human demonstrations. Along the way it measured the facts the sim needs: a **(+10, +11) mm** placement offset (the exact error behind our earlier 14 mm miss), the **(−7.6, +3.3) mm** jaw-midpoint shift when the moving jaw closes, the grasp stall/load, and wrist-camera parallax.
2. Argon **caught a calibration error no closed-jaw method could see**: from three open-jaw photos it found the wrist roll was **~190° off** and the gripper opened far wider than the model — reprojection error **130 px → 6.5 px**.
3. The **twin → game** step is done by an **Antigravity managed agent running Gemini 4 Argon**: it built the Roblox level from Argon's own sim and passed the local Luau + parity gates (live run: 3.3 min). The game is a real multiplayer lobby (20 tables, spectators, score/streak, leaderboard) whose physics and success checker are the same as the twin's (442 shared test vectors, 4.97 mm parity).

Now, heres how to experience this project:

**1. Watch the demo video** (3 min) — the demo video is on the hackathon submission page. It shows Argon experimenting on the real arm (sped up, both cameras, its thoughts on screen), its sim next to reality, the Roblox game built from that sim, and the policy trained from it running with Argon.

**2. Run the pipeline yourself — keyless, $0** (2 min). Everything runs offline with fake Gemini, a MuJoCo twin and a mock robot (no keys, no robot, no GPU). If you're driving an agent, just have it run these; at a terminal, same thing:

```bash
uv sync                                          # or: python -m venv .venv && pip install -e .
python -m r2p run --task place_on_cell --dry-run --rounds 2
                                                  # ▶ the WHOLE loop offline: probe the bench -> twin -> expert data ->
                                                  #   label -> train (CPU tiny) -> evaluate -> gate -> deploy (mock)
python -m r2p argon --task place_on_cell --step-run new --dry-run --approve auto
                                                  # ▶ Argon's orchestrator: experiment -> sim -> game -> play ->
                                                  #   train -> evaluate -> deploy, with a fake Argon and a mock arm
python scripts/demo_up.py                         # the judges' dashboard on http://127.0.0.1:8800/demo
```

That's where you see *why a robotics team would want this*. **With a Gemini API key**, Gemini 4 Argon makes every decision itself — same tools:

```bash
echo GEMINI_API_KEY=... > .env
python scripts/argon/build_game.py --task place_on_cell --out runs/game --allow-provisional-twin
                                                  # Antigravity managed agent builds + verifies the Roblox level
python scripts/argon/tictactoe.py --frame-only <board_photo.jpg>   # Argon reads the board and picks the move
```

**3. (optional) Run it on a real arm** (an SO-101 follower + a phone as the scene camera + any USB wrist camera). Hand your agent this recipe ([`docs/USAGE.md`](docs/USAGE.md) has every step):

```bash
python scripts/robot/armd.py --allow-unverified     # the guarded arm daemon (owns the serial port)
python scripts/lead/phonecam.py                      # phone browser -> scene camera; then calibrate it
python scripts/lead/wristcam.py                      # wrist camera over HTTP
python scripts/lead/argon_step.py --turns 40         # ▶ Argon experiments on the real arm and writes its sim
python scripts/lead/after_step.py --run <run>        # Argon's sim -> managed-agent game build -> play data
python scripts/lead/ttt_armd.py                      # play tic-tac-toe against the robot (Argon is the brain)
```

Honest heads‑up: the **experimenting, calibration, twin → game and the Argon tic-tac-toe brain** reproduce cleanly on the real bench; the **learned policy on the real arm** was still being qualified at submission (see FAQ #2) — expect proof the pipeline works end to end in the twin, and real-arm pick-and-place driven by Argon.

**4. Read the FAQ** at the end of this README — maybe I've already answered your question!



## The thesis

**Real bench → accurate twin → game → data → policy → real bench.** Robot learning is starved for data, and simulators are only useful if they match reality. If a frontier model can **experiment on the real robot until its own simulation matches what the cameras see**, then that twin can be turned into a **game anyone can play**, and every game round becomes **training data for the real robot**. Gemini 4 Argon is the scientist (experiments, critique, sim edits), the game designer (managed agent), and the brain at deploy time (high-level commands); a small learned policy does the 30 Hz control.

The honest version of the claim, in numbers: **a real pick-and-place learned by experimenting in 23 Argon turns, no human demonstration**; a calibration bug found by sight (**130 px → 6.5 px**); the trained state policy reaches **96.7 % clean / 80 % at half real noise / 43 % at full real noise in the twin (n = 30)**.

## Three legs

### Leg 1 — Real → twin (Gemini 4 Argon experimenting on hardware)
- **Calibration by sight**: an ArUco board placed flush against the robot base doubles as the ruler (no tape measure) → scene camera at **0.70 px** reprojection; Argon points at the jaw tips in 14 poses → joint offsets **84 px → 7 px**, leave-one-out **8.8 px (~5 mm)**; open-jaw frames → wrist roll fixed (**130 px → 6.5 px**).
- **One move at a time** (`scripts/lead/argon_step.py`): every turn Argon sees the scene camera, the wrist camera, the measured jaw pose, gripper/joint loads and a render of *its own* sim from the same calibrated camera, chooses **one** action (move / gripper / done), and edits its sim JSON (objects, sizes, colours, masses, frictions, arm offsets, servo gains). Full transcript every turn (1M context), low thinking, hard safety floor in the harness.
- **Argon Lab** (`r2p/lab/`): the same idea as an HTTP harness with primitives (`goto`, `grasp`, `place`, `move_by`, `look_at`, `deproject`) after Google's *Gemini 4 Argon for robotics* pattern, real ↔ sim replays of every recorded episode, and held-out episodes that gate every sim edit.

### Leg 2 — Twin → game → data (Antigravity managed agent + Roblox)
The game is built from Argon's twin by an **Antigravity managed agent running Gemini 4 Argon** ([`scripts/argon/build_game.py`](scripts/argon/build_game.py)): it receives the measured task, the physics and the harness sources, generates the level, and must pass the Luau unit tests and numeric parity in the sandbox **and again locally** before anything is accepted; it never publishes on its own. The Roblox place ([`roblox/`](roblox/)) has a lobby of 20 tables, multiplayer + spectators, score/streak, a leaderboard and authenticated trace upload. Play traces come back through `python -m r2p loop --from-game`: validated, **replayed in the twin**, relabelled into the robot's absolute joint-target chunks with Argon's command segments, and mixed into training.

### Leg 3 — Data → policy → real (B200 + Gemini 4 Argon at runtime)
A command-conditioned policy (Argon issues `pick(piece)` / `place_at(piece, cell)`; the policy outputs absolute 15×6 joint-target chunks at 30 Hz) is trained on a **B200 rented through our landlord GPU broker** (192 cores for parallel expert traces). The expert runs inside the **real arm's envelope** (calibrated limits, real start pose, real roll window): 1/64 → **63/64** after fixing over-squeeze, grasp height and fixed-jaw sweep. At deploy time **Argon reads the board and picks the move**, a non-blocking Argon cap sensor tracks the piece (white caps on white paper defeat colour segmentation), and the guarded daemon refuses anything outside the envelope.

## Run it

**Safe by default — offline.** The full loop runs end-to-end with a fake Argon, a MuJoCo twin and a mock robot:

```bash
python -m r2p run --task place_on_cell --dry-run --rounds 2
python -m r2p argon --task place_on_cell --step-run new --dry-run --approve auto
python -m r2p status
```

**Go live.** Put `GEMINI_API_KEY` in `.env` and start the bench services — the same tools hit the real model, arm and cameras:

```bash
python scripts/robot/armd.py --allow-unverified
python scripts/lead/argon_step.py --turns 40
python -m r2p argon --task place_on_cell --step-run <run> --approve ask
```

Setup, calibration, the bench services, Roblox publishing and the GPU recipe are in [`docs/USAGE.md`](docs/USAGE.md).

## Weights & models

| Model | Where | Used for |
|---|---|---|
| **Gemini 4 Argon** (`gemini-4-argon`) | Gemini API, Interactions API | Every decision: experiments, sim edits, calibration pointing, board reading, cap finding, orchestration |
| **Antigravity managed agent** (`antigravity-preview-09-2026`, model `gemini-4-argon`) | Gemini API | Builds and verifies the Roblox level from the twin in a Google-hosted sandbox |
| **SO-101 MJCF** | [TheRobotStudio SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) (see `assets/so101/ATTRIBUTION.txt`) | The arm in the MuJoCo twin |
| **State policy** (ours) | `exports/` (`r2p/policy`) | 3-seed MLP ensemble, absolute 15×6 chunks at 30 Hz |
| **SmolVLA** (fine-tuned, ours) | [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) + `r2p/vla` | Two-camera VLA conditioned on Argon's commands (data-limited, see FAQ #3) |

## Repo map

| Path | What |
|---|---|
| [`scripts/lead/argon_step.py`](scripts/lead/argon_step.py) | **Argon experimenting on the real arm**, one move per turn, building its sim |
| [`r2p/lab/`](r2p/lab/) | **Argon Lab** — primitives harness, editable MuJoCo sim workspace, real ↔ sim replays, held-out gate |
| [`r2p/orchestrator.py`](r2p/orchestrator.py) | **Argon orchestrator** — experiment → sim → game → play → train → evaluate → deploy as gated tools |
| [`scripts/argon/`](scripts/argon/) | Managed-agent game builder, task author, visual calibration, live QA |
| [`roblox/`](roblox/) | The Roblox game (lobby, tables, levels, Luau checker, trace upload) |
| `r2p/sim/` | MuJoCo twin with an emergent-sag servo model, experts, bench-profile expert |
| `r2p/policy/` · `r2p/vla/` | State policy training/eval/runtime · SmolVLA fine-tune |
| `r2p/robot/` | Guarded streaming arm daemon, policy runner, Argon cap perception |
| `r2p/perception/` · `r2p/real2sim/` | Calibrated cameras, pose fitting, twin critique/patch loops |
| `r2p/ttt/` · `scripts/lead/ttt_armd.py` | Argon's tic-tac-toe brain + real-arm play |
| `dashboard/` | The judges' demo app (`/demo`) |
| [`docs/USAGE.md`](docs/USAGE.md) | How to set up and run everything |

## Credits & responsible use

Built on **Gemini 4 Argon** (the only model making decisions; Gemini 3.8 Flash is disabled unless the owner opts in) and **Antigravity managed agents**, **MuJoCo**, **LeRobot**/SmolVLA, the **SO-101** open arm, and **Roblox**. GPUs were rented through a lease ledger so no job ever ran on a card without a recorded purpose. Every real-arm action passes through a guarded daemon (calibrated joint limits with a 2° margin, table-clearance checks, live abort on jaw height, roll-excursion limit); autonomous real-arm operation ran only with the owner present and authorizing it.

# FAQ

1. Did Gemini really do the experimenting, or did you script it? Argon did. Each turn it gets the two camera images, the jaw pose and its sim render, writes one thought and one action; the harness only executes and enforces safety. The full transcripts are logged (`runs/argon_step/<run>/ledger.jsonl`) and the demo video shows its thoughts on screen.
2. Is the learned policy running on the real arm? The pipeline is complete and the policy trains on a B200 from real-envelope expert data, but the expert only became real-arm compatible late in the hackathon (63/64 after the fix), so real-arm qualification of the learned policy was still in progress at submission. The real-arm pick-and-place you see is Argon itself (experimenting) and Argon + calibrated IK (tic-tac-toe); the learned policy is shown in the twin with honest numbers (96.7 % clean, 43 % at full real noise).
3. Why is the VLA weak? SmolVLA was fine-tuned on 400 twin-rendered episodes in 25 minutes and scores 1/30 in the twin; it needs more data and the same real-envelope fix. The state policy is the one that works.
4. Is the sim actually accurate? Kinematics and calibration yes (camera 0.70 px, joint offsets ~5 mm, the measured placement offset is applied); visually the quick CPU render Argon compared against was imperfect, and its sim JSON is a measured model of this bench rather than a photoreal clone. Twin object IoU on real frames: 0.84 / 0.95.


# Thats it! Thank you so much for hosting this, and I hope this was a cool project to you guys! :)
