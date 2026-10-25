# Agent C2: cameras, perception, synthetic bench, composite renderer

Read docs/briefs/COMMON.md first.
Own: r2p/perception/, r2p/synthbench/, tests/perception/, scripts/perception/. Use r2p.model, r2p.tasks (A),
r2p.gemini (D; import lazily, use a fake detector in tests).

Deliver (SPEC 6.7 perception + section 13):
1. r2p/perception/camera.py: CameraModel (K, dist, R, t in the spec frame; project; deproject onto the table plane
   z=h; JSON save/load with image size); open_camera(cfg) with LiveCamera (cv2 index/backend, background thread),
   HttpCamera (polls http://127.0.0.1:8792/frame/{index}.jpg, resizes to cfg size, age_s), StillCamera, SynthCamera.
2. scripts/perception/camserver.py: serves every attached camera (cv2 AVFoundation on macOS, default elsewhere) at
   1280x720 as /frame/<i>.jpg and /list JSON on 127.0.0.1:8792; plus a camserver.command launcher for macOS (must be
   launched from Terminal for camera permission).
3. r2p/perception/aruco.py + scripts/perception/calib_camera.py + make_board.py: GridBoard DICT_4X4_50 4x3, 40 mm
   markers, 10 mm gaps (configurable); intrinsics from 8-12 tilted views; extrinsics from the board lying flat at a
   measured placement relative to the base (top row toward the robot; offsets from config); report reprojection px
   and elevation; write config/camera.json; overlay image of the base footprint for checking; board PNG (US Letter,
   300 dpi, 100 %).
4. r2p/perception/appearance.py + task_perception.py: TaskPerceiver per INTERFACES. setup(frame): ONE SMALL-role
   Gemini call (r2p.gemini.call(role="small")) returns boxes for every task object; GrabCut inside each box + the task
   colour; per-object Lab histogram model (L 8 x a* 32 x b* 32) fitted against the rest of the frame (threshold fitted
   on that frame); tracking at 30 Hz by back-projection + nearest component; online background learning of whole
   distractor blobs on a confident track, rolled back if the object would lose > 10 % of its pixels; poses:
   hull-over-yaw for boxes/sticks (sticks yaw in [-90, 90)), rim circle for bowls, FK attach while held, stacked
   objects on their support's top; known-pose objects (task "xy" / fixed) never tracked; lost > 1 s -> async
   MAIN-role re-detect (<= 6 per run; 0 when replaying a saved setup); save_setup/load_setup (zero requests on replay;
   honour R2P_CACHE_ONLY).
5. r2p/synthbench: World(task, rng, cam) rendering the official arm (MuJoCo, spec frame) + table + ArUco board + any
   task's objects (screwdriver = stick with a yellow handle), rigid attach in the gripper frame when the jaws close
   within 2.5 cm, release onto container floor / target top, finger-pad push, truth poses, truth_check
   (r2p.tasks.check). Plus a composite renderer prototype (SPEC section 13): render arm+objects over a real background
   photo with a shadow catcher, tunable albedo/light/exposure/white balance/noise params, and a metric
   (masked Lab error + IoU) vs a real frame.
6. tests/perception: deprojection accuracy (synthetic, < 2 mm), board calibration on rendered views (camera position
   error < 1 cm), TaskPerceiver accuracy on synthbench scenes with oracle boxes (cube/bowl/stick < 3 mm), re-detect
   logic, setup save/load with zero requests, composite renderer runs.

Acceptance: `pytest -q tests/perception` passes.
