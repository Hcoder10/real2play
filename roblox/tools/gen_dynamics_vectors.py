"""Small deterministic FK / mass / Coriolis / gravity oracle from official MuJoCo."""
from pathlib import Path
import sys
import numpy as np
import mujoco
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from r2p.frames import MJCF, HOME_Q, BASE_ORIGIN_MJCF, FIXED_TIP, MOVING_TIP
from gen_presets import lua

def main():
    m=mujoco.MjModel.from_xml_path(str(MJCF)); d=mujoco.MjData(m)
    rng=np.random.default_rng(20261025); cases=[]
    for i in range(12):
        q=HOME_Q.copy() if i==0 else rng.uniform(m.jnt_range[:,0]*.8,m.jnt_range[:,1]*.8)
        v=rng.uniform(-.5,.5,6); d.qpos[:]=q; d.qvel[:]=v
        mujoco.mj_forward(m,d); mass=np.empty((m.nv,m.nv)); mujoco.mj_fullM(m,d,mass)
        def point(tip):
            b=m.body(tip[0]).id
            return d.xpos[b]+d.xmat[b].reshape(3,3)@tip[1]-BASE_ORIGIN_MJCF
        cases.append(dict(q=q.tolist(),v=v.tolist(),mass=mass.tolist(),bias=d.qfrc_bias.tolist(),
                          armature=m.dof_armature.tolist(),fixed=point(FIXED_TIP).tolist(),moving=point(MOVING_TIP).tolist()))
    path=Path(__file__).resolve().parents[2]/"tests/roblox/dynamics_vectors.luau"
    path.write_text("-- Generated MuJoCo oracle.\nreturn "+lua(cases)+"\n",encoding="utf-8")

if __name__=="__main__":main()
