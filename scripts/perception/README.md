# C2 perception and synthetic bench

All images are **uint8 BGR**; positions are **spec-frame metres**. Extrinsics map
spec coordinates into the OpenCV camera: `camera_xyz = R @ spec_xyz + t`.
`CameraModel.image_size` is `(width, height)`. Resize the calibration when resizing
frames. `read()` returns an image; `age_s` includes the HTTP server's capture age.

From the repository root on Windows:

```powershell
.venv/Scripts/python.exe -m pytest -q tests/perception
.venv/Scripts/python.exe scripts/perception/make_board.py
.venv/Scripts/python.exe scripts/perception/calib_camera.py --help
```

On macOS use `.venv/bin/python`. Open `camserver.command` in **Terminal** so macOS
can grant camera permission. Alternatively run `zsh scripts/perception/camserver.command`.
The server listens only on `127.0.0.1:8792`; `/list` lists cameras and
`/frame/<index>.jpg` serves 1280×720 images. `--indices 0 2` selects devices;
default discovery scans 0–9, including holes (`--scan N` changes that limit).

Print `board_letter.png` on US Letter at **100% / actual size**. Its PNG metadata
is 300 dpi. Measure a printed marker; defaults are 40 mm markers, 10 mm gaps,
4 columns × 3 rows, DICT_4X4_50. Top row faces the robot. Set measured placement
in a copy of `board_config.example.json`: `origin_xyz` is marker 0's top-left
black corner, relative to the base footprint centre. Columns run +y and rows +x.
Alternatively supply `base_front_x_m`, `dist_from_base_front_m`,
`left_edge_y_m`, and optionally `dist_from_base_front_right_m` for board yaw.
The front-face convention requires an explicit measured `base_front_x_m`.

Capture 8–12 tilted views (SPACE saves; q exits), then put the board flat at
its measured location and capture `flat.png`:

```powershell
.venv/Scripts/python.exe scripts/perception/calib_camera.py --config measured.json --capture views
.venv/Scripts/python.exe scripts/perception/calib_camera.py --config measured.json --intrinsic-views views --flat flat.png
```

The second command writes `config/camera.json` and
`config/camera_footprint.png`. Inspect the footprint and +x/+y overlay. The command
reports reprojection RMS, camera position, height and elevation. `--intrinsics`
reuses a calibration instead of refitting it; omit `--flat` to read the camera.
No camera calibration file is supplied as if it were a measured bench calibration.

```python
from r2p.perception import CameraModel, TaskPerceiver, open_camera
from r2p.tasks import load_task

cam = CameraModel.load("config/camera.json")
task = load_task("cube_in_bowl")
with open_camera(camera_config) as source, TaskPerceiver(cam, task) as tracker:
    tracker.setup(source.read())       # exactly one SMALL detector call
    tracker.save_setup("setup.json")
    result = tracker.estimate(source.read(), measured_q)
    success = tracker.check(result)
```

For replay, create a new tracker and call `load_setup("setup.json")`, then
`estimate` directly. Replay never calls a detector, including recovery. A live
run recovers objects lost for >1 second asynchronously with MAIN, at most six
calls and at least eight seconds between attempts. `R2P_CACHE_ONLY=1` disables
async recovery and the shared Gemini client enforces cache-only setup. Detector
failures are exposed in `redetect_error`; low-confidence missing poses have
`seen=False, source="stale"` and are excluded from the success checker.

Injected detectors implement `detect(frame, task, *, role)` and return
`{object_id: [x_min, y_min, x_max, y_max]}` in pixels. `OracleDetector(world)` is
for synthetic tests only and supplies boxes, not masks or poses. Known `xy`
objects and fixed objects are not colour tracked; fixed objects without a pose
are located once. Appearance models use Lab 8×32×32 likelihood histograms.
Size scaling is fitted only within a task's declared `randomize.size_scale`.

```python
import numpy as np
from r2p.synthbench import World, OracleDetector, CompositeRenderer, RenderParams

with World("cube_in_bowl", np.random.default_rng(2), cam) as world:
    image = world.render()             # CPU, official MuJoCo arm transforms
    world.step(absolute_q)             # one 30 Hz synthetic tick
    truth = world.poses()
    success = world.truth_check()      # the shared r2p.tasks.check
    renderer = CompositeRenderer(world, "empty_workspace.jpg",
                                 RenderParams(exposure=.2, noise_std=1))
    composite = renderer.render(q=recorded_q, poses=recorded_poses)
    metrics = renderer.compare(real_frame, real_object_masks, composite)
```

The synthetic bench is a deterministic perception fixture, not `r2p.sim`'s
dynamic reference twin. It rigidly attaches an object in the gripper frame on
closing within 2.5 cm, limits jaw closure to the object's width, releases onto
container floors/support tops, and models finger-pad pushes. The CPU renderer
clusters official mesh vertices at 0.7 mm and uses painter ordering; overlapping
geometry and grazing views can show rasterization artifacts. It does not create
a GPU context. Composite parameters cover RGB albedo, roughness, light,
ambient, shadows, exposure, white balance, blur, noise and JPEG quality; metrics
report per-object and mean mask IoU, masked CIE Lab error and combined loss.

Validation is synthetic/offline. Real-camera accuracy, real lighting robustness,
hardware access, Gemini quota/network integration, and VLA transfer require bench
validation. The renderer is a prototype, with an approximate shadow catcher.
