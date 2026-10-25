# Z1 bowl policy: qualification in progress

The 16:20 deadline was missed. No trained bowl export is currently qualified for hardware. No hardware motion or Gemini requests were made.

## Expert and scene

Z2's measured scene supersedes the nominal estimate: both brown cubes are on the robot's left, at (0.189677, 0.107700) and (0.249978, 0.080208) metres; the blue bowl is at (0.174270, 0.009673). `Z1_BOWL_TASK.json` preserves measured geometry and randomization, with a 240-second episode budget.

The free-roll, physical-contact expert passes the full gate with the corrected cube gripper profile: **181/192 successes (94.3%), zero target-limit violations**. Seeds are 5000-5063 and 6000-6127. Evidence: `out/z1/v3_expert_acceptance.json` and its two source reports. `Z1_BOWL_PROFILE_V3.json` uses the original real start, calibrated bounds, roll within 85 degrees of rest, and 0.5-radian release. Trace metadata derives the contact threshold (~0.4502 radians). Home preserves that release opening. There are no welded or teleported grasps.

The original front-layout probe passed 138/192; the first measured-layout profile passed 187/192. Its gripper threshold incorrectly classified cube contact as open, so that profile is retained only as diagnostic evidence.

## Trained exports

| Export | Successful traces | Training | Guarded measured-scene rehearsal |
| --- | ---: | --- | --- |
| `exports/bowl_v1` | 169 | 60000 updates/member, seeds 0/1/2, batch 256, width 512, 3 blocks, no augmentation | Failed; 60 refusals, start accepted, zero sent owner-limit violations |
| `exports/bowl_v3` | 36 | Same recipe, corrected gripper convention | Failed; 78 refusals, start accepted, zero sent owner-limit violations |

Both exports declare `qualification.ready_for_real=false`; `run_bowl_policy.py --real` rejects them before opening hardware. Original v1 held-out evaluation was 2/30 clean, 0/30 real noise, 0/30 half noise. Those results used the evaluator's legacy 0.10-radian step, which did not match the bench's 0.002327-radian step; they are not bench-rate qualification. The corrected v3 four-scene bench-rate probe is 0/4. Earlier results remain available as failure evidence.

The larger 2304-episode corrected collection is running on the B200 with 190 processes, seeds 20000-22303. Successful traces are being backed up to `out/z1/v3_trace_backup/` in SHA-256-verified archives as collection proceeds. Full-data training is queued as `bowl_v4` after its completed report. A 256-trace interim fit also tests a lower final learning rate; that optimizer variation is explicitly recorded.

## Fixes and validation

- Separate `r2p/sim/bowl_expert.py`: free-roll grasp branches and open-jaw home.
- Export/training support preserves the free-roll window and padded observation slots.
- NumPy GELU retains float32, avoiding unnecessary float64 inference work.
- Twin evaluation now honors the export's deployment joint-step limit; legacy exports retain 0.10.
- Saved vision setup accepts the exported gripper threshold, preserving existing callers' default.
- Bowl runner audits exact calibrated bounds and start, rehearses with the real Session/Guard, and requires explicit bus release for direct hardware access.

Acceptance command (26 passed):

```powershell
.venv/Scripts/python.exe -m pytest -q tests/sim/test_bowl_profile.py tests/policy/test_bowl_export.py tests/policy/test_bench_export.py tests/policy/test_temporal_context.py tests/policy/test_policy.py tests/robot/test_bench_hardening.py tests/perception/test_bowl_gripper.py
```

## Reproduce and operate

All B200 GPU commands require the lead's active lease environment (`apt_8fc32d1f`, GPU 0, 170 GB cap), plus `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and `R2P_CACHE_ONLY=1`. The rental expires at 17:30; its shared lease must not be released by Z1.

```sh
python3 scripts/robot/z1_collect.py --task scripts/robot/Z1_BOWL_TASK.json --profile scripts/robot/Z1_BOWL_PROFILE_V3.json --out out/z1/v3_collection --n 2304 --workers 190 --seed 20000 --save
python3 scripts/robot/z1_train.py --task scripts/robot/Z1_BOWL_TASK.json --traces out/z1/v3_collection/success --out exports/bowl_v4 --steps 60000 --seeds 0 1 2 --workers 24
python3 scripts/robot/z1_eval.py --export exports/bowl_v4 --n 30 --workers 4 --noise clean
python3 scripts/robot/z1_eval.py --export exports/bowl_v4 --n 30 --workers 4 --noise real
python3 scripts/robot/z1_eval.py --export exports/bowl_v4 --n 30 --workers 4 --noise realx0.5
python3 scripts/robot/run_bowl_policy.py --export exports/bowl_v4 --bench-profile scripts/robot/Z1_BOWL_PROFILE_V3.json --rehearse --timeout 240 --record out/z1/v4_rehearsal
```

Local Python is `.venv/Scripts/python.exe`. The eventual live command below is **blocked for the current unqualified export**. It is documented for handoff, not a recommendation to execute:

```sh
python3 scripts/robot/run_bowl_policy.py --export exports/bowl_v1 --task tasks/bowl_bench/task.json --bench-profile scripts/robot/Z1_BOWL_PROFILE_V3.json --perception-setup runs/bench/bowl_perception.json --real --transport direct --arm-released --allow-unverified --timeout 240 --record runs/robot/bowl_v1
```

## Demo evidence

Local MP4s are in `out/demo/bowl_sim/`, calibrated with `config/camera.json`, MuJoCo EGL, 640x360 at 10 fps, explicitly captioned SIMULATION and 3x playback. Three initial expert demonstrations and v1 policy success/failure episodes are complete and decoded successfully. V3 early-policy failure videos are also present; corrected expert video is transferring. The v1/v3 early policy clips use the legacy evaluator rate and must be described as such. A success clip does not establish a reliable policy.

Pending: full corrected data collection/training, final clean/noisy evaluations, successful guarded qualification, and remaining artifact transfer. The old `out/z1/measured_probe.tar.gz` is a partial transfer, not a usable archive.
