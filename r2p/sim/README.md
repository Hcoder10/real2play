# Physical reference simulator

```python
from r2p.tasks import load_task, sample_scene
from r2p.sim import TaskTwin, gen_traces
from r2p.sim.expert import Expert
import numpy as np

task = load_task("place_on_cell")
scene = sample_scene(task, np.random.default_rng(10000))
with TaskTwin(scene) as twin:
    result = Expert(task).run(twin, seed=10000)
    print(result["meta"]["success"], result["check"])

paths = gen_traces(task, 10, "traces/cell", workers=1, seed=0, dart=.01)
```

The world origin is the base footprint on the table, in metres. The official
MJCF is translated once by `-BASE_ORIGIN_MJCF`. `step(q_target)` takes six
absolute joint targets and advances one 30 Hz control tick; the calibrated
configuration runs 32 internal steps. `q()`, `poses()`, `grip_closed()` and
`render(camera)` implement the shared interface. A camera may be an ID from the
scene or a dictionary containing pos/lookat/fovy_deg/width/height. Use the context
manager or `close()` to release render resources.

The simulator applies finite-stiffness, torque-limited position servos to the
official masses/inertias. Gravity sag is physical. Jaws use narrow collision pads
on the actual finger bodies and moving hinge. Objects are free bodies unless
the scene says fixed; no attachment, teleport, hidden success rule, or proximity
grasp is used. Bowls contain a disc and sixteen wall staves. Sticks are square
prisms; non-colliding plates are visual targets.

The expert uses damped jaw-midpoint IK with downward orientation and wrist-roll
control in the null space, measured-state feedback, and gravity feedforward.
It grasps at low fingertip clearance, measures the held object's offset after
carrying/rotating, releases to 0.5 rad, and retreats. Pushes stop from observed
object progress; the closed fist backs off before opening. Skill commands label
each segment. DART changes executed targets only. Trace metadata preserves the
scene, physics snapshot/hash, start pose, command intervals and actual outcome.

Acceptance commands (use `.venv/Scripts/python.exe` on Windows):

```sh
python -m pytest -q tests/sim tests/policy
python -m r2p.sim.expert --task place_on_cell --n 100
python -m r2p.sim.expert --task cube_in_bowl --n 100
```

`--out path.json` saves a report with scene seeds, success counts, failure
reasons, placement error, and task/physics/code hashes. `expert_results.json`
contains the recorded acceptance results. All success decisions use
`r2p.tasks.check_window` on the final 0.5 seconds. These are simulation results;
the provisional servo parameters still need held-out real-arm identification.

Recorded acceptance (physics version 2, seeds starting at 10000):

| Task | Success | Median placement error |
|---|---:|---:|
| place_on_cell | 100/100 | 1.97 mm |
| cube_in_bowl | 95/100 | — |
| pick_place_stick | 10/10 | 1.54 mm |
| stack_two_cubes | 10/10 | — |
| push_to_target | 10/10 | 5.87 mm |
| two_cubes_in_bowl | 10/10 | — |

The component suite passed all 18 tests, including both physical grasp holds,
sag bands, six-task regressions, trace validation/DART, multiprocessing collection,
tiny CPU training, two-seed export, standalone runtime parity (1e-4), and all
three evaluation noise modes. These expert success rates do not describe the
tiny trained smoke policy; no deployment-quality learned incumbent was trained.
