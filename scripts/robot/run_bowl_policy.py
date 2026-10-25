"""Bowl policy audit, physical twin rehearsal, or explicitly requested live run."""
import _bootstrap
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from r2p.robot.config import load_config, load_joint_map, config_path
from r2p.robot.rehearsal import (audit, make_session, validate_profile, TwinRobot, TwinWorld,
                                require_released_bus, streaming_robot)
from r2p.robot.operator import Operator
from r2p.sim.bench import load_profile
from r2p.sim.twin import TaskTwin
from r2p.tasks import load_task, sample_scene
from r2p.model import Arm

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export',default='exports/bowl_v1'); p.add_argument('--task',default='tasks/bowl_bench/task.json')
    p.add_argument('--bench-profile',default='scripts/robot/Z1_BOWL_PROFILE.json'); p.add_argument('--config')
    mode=p.add_mutually_exclusive_group(); mode.add_argument('--real',action='store_true'); mode.add_argument('--rehearse',action='store_true')
    p.add_argument('--transport',choices=['direct','armd'],default='direct'); p.add_argument('--arm-released',action='store_true')
    p.add_argument('--allow-unverified',action='store_true'); p.add_argument('--timeout',type=float,default=240)
    p.add_argument('--perception-setup',default='runs/bench/bowl_perception.json')
    p.add_argument('--record',type=Path,default=Path('out/z1/rehearsal')); p.add_argument('--seed',type=int,default=0)
    a=p.parse_args()
    from r2p.policy import load
    policy=load(a.export); cfg=load_config(a.config); profile=load_profile(a.bench_profile)
    if a.real and policy.config.get('qualification',{}).get('ready_for_real') is False:
        raise ValueError('Export failed qualification: '+policy.config['qualification'].get('reason','see its recorded rehearsal'))
    task=load_task(a.task); joint_map=load_joint_map(cfg); limits=validate_profile(profile,joint_map)
    if task['name']!='bowl_bench': raise ValueError('Expected bowl_bench')
    if a.timeout<=0 or not np.isfinite(a.timeout): raise ValueError('timeout must be positive and finite')
    if a.rehearse and a.transport!='direct': raise ValueError('Bowl rehearsal currently exercises the direct transport only')
    diagnostics=audit(policy,task,profile,joint_map,cfg)
    print(json.dumps({'motion':False,'audit':diagnostics},indent=2),flush=True)
    a.record.mkdir(parents=True,exist_ok=True)
    (a.record/'audit.json').write_text(json.dumps(diagnostics,indent=2))
    if not diagnostics['start_valid']: return 1
    if not a.real and not a.rehearse: return 0
    if a.rehearse:
        # Frozen measured scene at seed 0; optional seeded randomized rehearsal.
        frozen=deepcopy(task)
        if a.seed==0: frozen['randomize']={'min_sep_m':0}
        scene=sample_scene(frozen,np.random.default_rng(a.seed)); arm=Arm()
        rest=joint_map.to_model(profile['owner_state']['rest'])
        with TaskTwin(scene,physics=cfg['rig'],q0=rest,joint_limits=limits.model_bounds(joint_map)) as twin:
            robot=TwinRobot(twin,joint_map,limits)
            world=TwinWorld(twin,task,arm,policy.config['gripper_open_threshold_rad'])
            session=make_session(robot,policy,task,profile,cfg=cfg,arm=arm,world=world,realtime=False,
                                 operator=Operator(None,keyboard=False),record=a.record/'trace')
            result=session.run(timeout_s=a.timeout)
            result.update(audit=diagnostics,final_poses=twin.poses(),synthetic=True,
                          sent_owner_limit_hits=sum(int(np.any(limits.violations(q))) for q in robot.sent))
    else:
        if policy.config.get('smoke_only'): raise ValueError('Smoke-only export cannot run hardware')
        if not joint_map.verified and not a.allow_unverified: raise ValueError('Use --allow-unverified for the calibrated half-speed convention')
        from r2p.perception import CameraModel, open_camera
        from r2p.robot.policy_vision import load_placement_setup
        from r2p.robot.io import RealRobot
        from run_real_policy import checked_rest
        if a.transport=='direct': require_released_bus(cfg,a.arm_released)
        camera=open_camera(cfg['camera']); robot=None; perceiver=None
        try:
            calibration=CameraModel.load(config_path(cfg['camera']['calibration']))
            perceiver=load_placement_setup(calibration,task,a.perception_setup,
                gripper_open_threshold_rad=policy.config['gripper_open_threshold_rad'])
            initial_q=joint_map.to_model(profile['owner_state']['rest'])
            poses=perceiver.estimate(camera.read(),initial_q)['poses']
            for obj in task['objects']:
                pose=poses.get(obj['id'],{})
                if not pose.get('seen') or pose.get('source')!='vision': raise ValueError('Object not freshly visible: '+obj['id'])
                if np.linalg.norm(np.asarray(pose['xyz'])[:2]-np.asarray(obj['xyz'])[:2])>.025: raise ValueError('Object moved outside measured layout: '+obj['id'])
            if a.transport=='direct':
                require_released_bus(cfg,a.arm_released); checked_rest(cfg,profile,joint_map); robot=RealRobot(cfg)
            else: robot=streaming_robot(cfg,profile,joint_map)
            with Operator(cfg['control_file'],keyboard=True) as operator:
                session=make_session(robot,policy,task,profile,cfg=cfg,camera=camera,perceiver=perceiver,
                                     realtime=True,operator=operator,record=a.record/'trace')
                result=session.run(timeout_s=a.timeout)
        finally:
            if perceiver is not None: perceiver.close()
            camera.close()
            if robot is not None: robot.close(torque_off=False)
    (a.record/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    return 0 if result.get('success') and result['outcome']=='goal_reached' else 1

if __name__=='__main__': raise SystemExit(main())
