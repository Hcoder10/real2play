# Agent C1: real-arm stack

Read docs/briefs/COMMON.md first.
Own: r2p/robot/, config/bench.json, config/bench.darwin.json, config/joint_map.json, tests/robot/, scripts/robot/.
Use r2p.model (Arm, JointMap), r2p.tasks (A), r2p.policy.load (B, lands later), r2p.perception (C2, lands later):
import lazily, depend on the INTERFACES names.
Bench = macOS Mac mini (Python 3.12, lerobot[feetech]==0.6.1); follower on /dev/tty.usbmodem5B3D0496341, robot id
follower_1, calibration dir configurable. Scene camera via camserver http://127.0.0.1:8792/frame/<i>.jpg (C2).
SPEC 6.7 and section 9 pitfalls are binding.

Deliver:
1. r2p/robot/io.py: RealRobot (LeRobot SOFollower, use_degrees; torque-on PRIMED: torque off -> read Present ->
   Goal := Present -> configure(); never drops torque except an explicit operator torque-off;
   disable_torque_on_disconnect False), ReadOnlyBus (no register writes; keep the SOFollower object alive),
   loads() via load_magnitude that handles LeRobot's already-signed Present_Load (-32 -> 32), MockRobot (first-order
   servo + gravity sag k * tau_g from Arm.gravity_torque; grip floor when holding).
2. r2p/robot/gravity.py: GravityComp(arm, k): target + k * tau_g(target) on lift/elbow/wrist_flex, capped 3 deg/joint;
   GravitySag for the mock. Default k from config (0.05).
3. r2p/robot/guard.py: joint limits minus 2 deg, per-tick step cap with a slow zone near the table, link/gripper/
   finger clearance vs the table plane (Arm.geometry), roll excursion cap; refusals HOLD with torque on; retreat moves
   never refused when they improve geometry.
4. r2p/robot/runner.py Session: 30 Hz loop; the policy obs uses the last COMMANDED q (reset to measured when any joint
   differs > 4 deg); ACT temporal ensembling (exp(-0.1 * age_rank), oldest weighted most); GravityComp before Guard;
   load stop; operator poll; stops: goal_reached (r2p.tasks.check on perceived poses holds 1 s, gripper open, jaw
   midpoint >= 7 cm up), policy_done, timeout, held_after_refusals (60), operator; start pose = export start_q;
   summary (outcome, timing p50/p90, refusals, success: truth on synthetic, perception check, operator y/n override).
   Record each attempt as r2p_trace/1 + frames/NNNN.jpg via a writer thread (bounded queue; drop frame AND row together).
5. r2p/robot/operator.py: keys s/SPACE stop (hold + slow retreat), p pause, g go, y/n score, q quit, T twice within 3 s
   torque off; POSIX cbreak keyboard + Windows msvcrt; control file runs/control.json {"seq", "cmd"}.
6. scripts/robot/: armd.py (one process owns the bus; HTTP 127.0.0.1:8793 GET /state, POST /cmd
   {joints|xyz|gripper|rest|release}; interpolated moves <= 20 deg/s with path clearance check and lag/load abort; xyz
   via damped IK fingers-down; release only at rest), preflight.py (read-only: port, 6 motors, EEPROM homing/ranges ==
   calibration file, supply 11-13.5 V, temps, torque, error flags, camera fps/size/exposure, camera calibration
   present, joint map verified flag, policy loads and acts, pytest; PASS/WARN/FAIL lines, exit 1 on FAIL), fit_sag.py
   (through armd: 8 supervised moves r .18-.30 m, left/centre/right, z .05/.10/.15; least squares of measured deflection
   vs -tau_g; --simulate K), run_policy.py (--policy, --task, --attempts, --speed, --dry-run on C2's synthbench,
   --record, --perception-setup, --gravity-comp, --allow-unverified), record_session.py (hand-guided, torque off),
   mirror.py, read_pose.py (touch fit of joint offsets).
7. config: bench.json (control hz 30, replan_every 1, ensemble_m 0.1, gravity_comp_k 0.05, policy_q commanded, guard
   params, camera {source http, url, index, width 640, height 360}, board, perception sizes), bench.darwin.json overlay
   (port, calibration_dir /Users/squaredcube/argon/calibration/robots/so_follower), joint_map.json with candidates
   identity and via_corpus_audit (offsets [0, -11.5, 0, 0, -76] deg), active via_corpus_audit, unverified; a loader
   that overlays bench.<sys.platform>.json.
8. tests/robot (no hardware): MockRobot dry runs with a scripted stub reach the goal with sag + comp; commanded-q
   prevents sag integration (show the delta-on-measured failure in a test); guard refusals hold; load_magnitude;
   operator control file; armd against MockRobot.

Acceptance: `pytest -q tests/robot` passes; `python scripts/robot/run_policy.py --dry-run --policy stub --task
cube_in_bowl --attempts 3` reaches the goal 3/3 (with C2's synthbench; until it lands, a MockRobot-only dry run with
truth poses).
