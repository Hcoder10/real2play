# SO-101 reference physics

`so101_bench.json` defines joint position servos in Nm/rad, damping in Nm·s/rad,
armature in kg·m², Coulomb friction and torque caps in Nm. `latency_ticks` delays
absolute targets at 30 Hz; each tick has 32 physics substeps (960 Hz). Object friction is
dimensionless and masses are kg. The initial values are a provisional calibration,
not a claim that the held-out real-twin accuracy gate has passed.

Sag is the static equilibrium of gravity, the official MJCF inertias, and finite
servo stiffness. There is no injected height displacement and no gravity
cancellation in `TaskTwin`. The expert compensates its targets, just as the real
runner does. `sag_gain` is the deployment-compatible compliance hint (rad/Nm);
the physics itself uses the individual `joints.*.kp` values.

Version 2 gives 3.26 mm static jaw sag at 0.18 m reach and 9.83 mm at 0.30 m
(jaw command z=0.08 m, gripper=0.3 rad, settled for four seconds). Elbow stiffness
is 30 Nm/rad; lift and wrist-flex stiffness are 20 Nm/rad. Consumers of the
compensation hint must preserve the six-vector: reducing it to one scalar loses
the fitted difference between short and long reach.

To refit with `r2p.sysid`, record commanded/measured joints and calibrated jaw
positions after at least two seconds at multiple safe poses spanning 0.18–0.30 m
reach, both empty and with known payloads. Fit static residual `q_command-q` to
MJCF gravity torque to estimate inverse stiffness, excluding saturated joints
and table contact. Use both approach directions to distinguish friction and
backlash. Fit small step-response trajectories for damping, armature and integer
latency; fit pushes/drops separately for object friction and contact softness.
Keep entire trajectories and unseen reaches out of the fit, then replay them and
record jaw RMS, object pose error, timing and grasp agreement before approving
a new version. Do not compensate the measured trace before fitting sag.

The fixed and moving jaw collision pads are narrow boxes following the official
finger geometry; the moving pad rotates on the actual gripper hinge. This avoids
the false solid mouth produced by a single convex hull of a non-convex jaw mesh.
All grasp forces come from MuJoCo contacts and Coulomb friction.

The 960 Hz step and 15 ms contact time constant resolve jaw and bowl impacts. A friction/normal impedance
ratio of 10 suppresses soft-contact creep without the no-slip postprocessor,
which introduced energetic release transients in the bowl tests. Regression
checks include thin-piece transport and two-second holds as well as bowl drops.

After changing this canonical JSON, the Roblox owner must regenerate its copied
preset with `python roblox/tools/gen_presets.py` and rerun parity. Policy exports
keep a physics snapshot and hash, so their recorded evaluations remain tied to
the reference version they actually used.
