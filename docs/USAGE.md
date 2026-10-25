# How to use Real2Play

## 1. Install

```bash
git clone https://github.com/Hcoder10/real2play && cd real2play
uv sync                      # or: python -m venv .venv && .venv/Scripts/pip install -e .   (Windows)
echo GEMINI_API_KEY=... > .env   # only for live Gemini 4 Argon; everything below has an offline mode
```

Python 3.12, MuJoCo 3.13. Real-arm driving also needs `lerobot[feetech]` (an env that has it works:
`PYTHONPATH=. <that-python> scripts/robot/armd.py`).

## 2. Offline (no keys, no robot, no GPU)

```bash
python -m r2p run --task place_on_cell --dry-run --rounds 2      # probe -> twin -> data -> label -> train -> eval -> gate -> deploy (mock)
python -m r2p status
python -m r2p argon --task place_on_cell --step-run new --dry-run --approve auto   # Argon orchestrator, fake Argon + mock arm
python scripts/argon/build_game.py --task place_on_cell --out runs/game --dry-run   # managed-agent game build, fake agent, real Luau gates
python -m pytest -q tests                                          # the test suite
```

## 3. The bench (real arm)

Hardware: SO-101 follower on a USB serial port, a phone as the fixed scene camera (browser page, no app), a USB wrist camera.

```bash
python scripts/robot/armd.py --allow-unverified    # start ONLY with the arm resting in its rest pose; owns the serial port
python scripts/lead/phonecam.py                     # open http://<laptop>:8792/ on the phone (USB tether is fastest)
python scripts/lead/wristcam.py --index 1           # wrist camera -> http://127.0.0.1:8794/frame/0.jpg
python scripts/lead/record_both.py --wrist-url http://127.0.0.1:8794   # record both cameras (runs/bench/video/)
```

Calibration (once per camera placement):
1. Scene camera: print the ArUco GridBoard (DICT_4X4_50, 4x3, 40 mm markers, 10 mm gaps), lay it flush against the robot
   base, run the camera calibration in `r2p/perception` -> `config/camera.json` (target < 1 px).
2. Arm: `python scripts/argon/visual_calib.py` (Argon points at the jaw tips in safe poses; open-jaw frames fix the wrist
   roll and gripper opening) -> a candidate in `config/joint_map.json`.
3. Board: `python scripts/lead/measure_grid.py --write` re-measures the printed grid by camera (do this whenever the paper moves).

## 4. Argon experimenting on the real arm -> sim -> game -> data -> policy -> deploy

```bash
python scripts/lead/argon_step.py --turns 40                         # Argon: one move per turn, both cameras, writes runs/argon_step/<run>/sim.json
python scripts/lead/after_step.py --run <run>                        # Argon's sim -> physics -> managed-agent game build -> play data -> twin replay
python -m r2p argon --task place_on_cell --step-run <run> --approve ask   # or let Argon orchestrate every stage (asks before any motion)
python scripts/lead/ttt_armd.py                                      # tic-tac-toe vs the robot (Argon reads the board, finds caps, places)
python scripts/robot/run_real_policy.py --export <policy> --argon-perception --transport armd \
    --bench-profile scripts/robot/K_STREAM_PROFILE.json --cap 0 --real --allow-unverified   # learned policy, Argon picks the cell
```

Bowl task: `python scripts/lead/argon_step.py --task two_cubes_in_bowl --goal "pick a brown box and put it in the blue bowl"`
(layout in `tasks/bowl_bench/task.json`).

## 5. Training on a GPU

Expert traces under the real arm's envelope, then the policy:

```bash
python -m r2p.sim.traces --task place_on_cell --bench-profile scripts/robot/K_BENCH_PROFILE.json --n 100 --seed 0 --out-dir traces/
python -m r2p.policy.train --task place_on_cell --traces traces/ --name ttt_policy --steps 15000 --batch-size 512 \
    --seeds 0 1 2 --no-augmentation --device cuda:0 --memory-fraction <fraction> --eval-n 6 --eval-max-ticks 4200
python scripts/robot/rehearse_policy.py --export ttt_policy --cell r1c1 --cap 0      # twin rehearsal under the real limits
```

Run many `r2p.sim.traces` processes in parallel (different `--seed`s) on a high-core box; keep a few cores free for the
trainer, which parses traces single-threaded.

## 6. Roblox game

```bash
python roblox/tools/gen_presets.py && python roblox/tools/test.py   # presets from the twin + Luau unit/parity tests
```

Build `out/roblox/Real2Play.rbxlx`, open it in Roblox Studio, **File > Publish to Roblox** to your place, enable
**Game Settings > Security > Allow HTTP Requests**, and set max players >= 30 (20 tables + spectators). Play traces
upload to the receiver started by `scripts/demo_up.py` (port 8770; expose it with a tunnel and put its URL and the key
from `out/demo/trace_key.txt` in the server-only `ServerStorage/LocalConfig`; never in a client script).

## 7. Dashboard

`python scripts/demo_up.py` -> http://127.0.0.1:8800/demo (Argon Lab / step-loop panel, chain, game, deploy, live cameras).
