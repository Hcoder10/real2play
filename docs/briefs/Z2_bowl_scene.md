# Agent Z2: measure the bowl bench and prepare the real deploy (deadline 16:15)
Repo C:\Users\sarta\real2play. The owner placed TWO brown 38 mm cubes and a large blue bowl (~14 cm) on the robot's LEFT
(camera-right in http://127.0.0.1:8792/frame/0.jpg; calibrated camera config/camera.json; spec frame metres, +x away from
the robot, +y robot-left). The tic-tac-toe board stays where it is (do not disturb it). Read-only camera GETs only; NEVER
move the arm (no POSTs to :8793). gemini-4-argon (r2p/gemini.py, MAIN, thinking low; see scripts/lead/ttt_armd.py
locate_caps for the pointing+deproject pattern) may be used to locate objects.
1. Locate both cubes (deproject the top-face centre at 38 mm) and the bowl (rim centre + diameter at its rim height) ->
   write tasks/bowl_bench/task.json from tasks/two_cubes_in_bowl/task.json: both cubes brown, measured positions/sizes,
   randomization +-2 cm around the measured spots (no overlap with the tic-tac-toe paper: its cells are in
   tasks/place_on_cell/task.json), success predicates unchanged. Validate with r2p.tasks.validate and sample 50 scenes.
2. A perception setup for the real run: r2p.perception.TaskPerceiver(camera, task).setup(frame) on a fresh frame (colour
   segmentation of brown cubes and the blue bowl) -> runs/bench/bowl_perception.json; check the estimated poses match
   your measurements (< 1 cm).
3. Report positions, the files, and an overlay image runs/bench/bowl_overlay.jpg (projected cube/bowl outlines on the
   real frame). Commit tasks/bowl_bench/task.json with an explicit path.
