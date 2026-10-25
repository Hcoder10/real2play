import json
from pathlib import Path
import numpy as np
from r2p.sim.bowl_expert import Expert
from r2p.tasks import load_task, sample_scene
from r2p.sim.twin import TaskTwin
from r2p.robot.rehearsal import validate_export
from r2p.policy.features import export_config
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]

def test_bowl_free_roll_stays_in_calibrated_window():
    profile=json.loads((ROOT/'scripts/robot/Z1_BOWL_PROFILE.json').read_text())
    task=load_task(ROOT/'scripts/robot/Z1_BOWL_TASK.json')
    expert=Expert(task,deployment=profile)
    assert expert.ik.roll is None
    np.testing.assert_allclose(expert.arm.range[4],profile['roll_window_rad'])
    assert profile['roll_window_rad'][0]>=profile['model_limits'][4][0]
    assert profile['roll_window_rad'][1]<=profile['model_limits'][4][1]
    assert np.max(np.abs(np.asarray(profile['roll_window_rad'])-profile['roll_rad']))<=np.deg2rad(85)+1e-9
    with TaskTwin(sample_scene(task,np.random.default_rng(0)),q0=profile['start_q']) as twin:
        expert._bind(twin)
        for obj in ('cube_a','cube_b'):
            yaw=expert._grasp_yaw(obj)
            roll=yaw+expert.qplan[0]+.06
            assert expert.arm.range[4,0]<roll<expert.arm.range[4,1]

def test_padded_bowl_slots_are_valid_for_real_runner():
    task=load_task(ROOT/'scripts/robot/Z1_BOWL_TASK.json')
    cfg=export_config(task,'bowl_test')
    assert cfg['K']==4 and len(task['objects'])==3
    validate_export(SimpleNamespace(config=cfg),task)

def test_bowl_home_preserves_open_release_above_contact_threshold():
    profile=json.loads((ROOT/'scripts/robot/Z1_BOWL_PROFILE_V3.json').read_text())
    task=load_task(ROOT/'scripts/robot/Z1_BOWL_TASK.json')
    expert=Expert(task,deployment=profile)
    with TaskTwin(sample_scene(task,np.random.default_rng(0)),q0=profile['start_q']) as twin:
        expert._bind(twin)
        expert.move=lambda *args,**kwargs:None
        targets=[]
        expert._tick=lambda q:targets.append(np.asarray(q).copy())
        expert.home()
    assert targets[-1][5]==profile['gripper_open_rad']
    assert targets[-1][5]>expert.deployment['gripper_open_threshold_rad']
