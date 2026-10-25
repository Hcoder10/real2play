# Agent X: 15-second sped-up demo clip of Gemini 4 Argon experimenting on the real arm (deadline 16:05)
Sources (read-only): the two-camera recording runs/bench/video/20261025_133414/ (scene_NNN.mp4 / wrist_NNN.mp4, 60 s
parts; scene_ts.txt / wrist_ts.txt lines "<part> <unix time>" per written frame; marks.txt "<unix time> <label>"),
and Argon's step-loop run runs/argon_step/step_01/: ledger.jsonl (one line per turn: {"turn","t" (unix time of the
reply), "obs":{jaw_xyz_m,gripper,...}, "reply":{"thought","action"}}), per-turn images t<NN>_real_vs_sim.jpg and
t<NN>_wrist.jpg. The step loop started at the marks.txt line "argon_step_start" (~15:2x local) and ran 23 turns ending
with a real pick of a white bottle cap and a clean place into the centre cell.
Make out/demo/argon_experiments_15s.mp4 (1920x1080 H.264, 30 fps, <= 15 s, no audio): scene camera large + wrist camera
inset, sped up so the whole 23-turn session (or its most motion-rich parts: approach, grasp, lift, carry, place) fits;
caption bar with "Gemini 4 Argon - experimenting on the real arm" + per-segment Argon thought snippets (short, from the
ledger) and the action (move/gripper); a turn counter; end card "Real pick-and-place learned by experimenting".
Use ffmpeg: C:/Users/sarta/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1-full_build/bin/ffmpeg.exe
(OpenCV for overlays is fine). Keep it fast: no re-encoding of the whole hour, only the window. Report the path and a
3-frame contact sheet path. Commit only the script you write (scripts/lead/make_demo_clip.py), not the video.
