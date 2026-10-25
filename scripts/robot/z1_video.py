"""Calibrated scene-camera MP4s from recorded expert or policy states (EGL)."""
import _bootstrap
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import mujoco
import numpy as np
from r2p.sim.twin import build_xml, load_physics
from r2p.frames import JOINTS
from r2p.perception import CameraModel
from r2p.traces import read_trace

class Recorder:
    def __init__(self, scene, physics, path, title, stride=9):
        self.stride=stride; self.title=title; self.frames=0
        self.cam=CameraModel.load('config/camera.json').resized((640,360))
        cam=self.cam
        root=ET.fromstring(build_xml(scene,load_physics(physics)))
        rotation=cam.R.T@np.diag([1,-1,-1]); quat=np.zeros(4)
        mujoco.mju_mat2Quat(quat,rotation.ravel())
        ET.SubElement(root.find('worldbody'),'camera',name='calibrated_scene',
                      pos=' '.join(map(str,cam.position)),quat=' '.join(map(str,quat)),
                      fovy=str(np.rad2deg(2*np.arctan(cam.height/(2*cam.K[1,1])))))
        self.m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
        self.d=mujoco.MjData(self.m)
        self.m.vis.global_.offwidth=640; self.m.vis.global_.offheight=360
        self.renderer=mujoco.Renderer(self.m,height=360,width=640)
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        self.writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10,(640,360))
        if not self.writer.isOpened(): raise RuntimeError('MP4 writer unavailable')
        # Map output distorted calibration pixels into the ideal centered GL camera.
        yy,xx=np.indices((360,640),dtype=np.float32)
        rays=cv2.undistortPoints(np.stack((xx,yy),-1).reshape(-1,1,2),cam.K,cam.dist).reshape(360,640,2)
        self.mx=(rays[:,:,0]*cam.K[1,1]+320).astype(np.float32)
        self.my=(rays[:,:,1]*cam.K[1,1]+180).astype(np.float32)

    def frame(self, tick, q, poses, command):
        if tick%self.stride: return
        for j,v in zip(JOINTS,q): self.d.qpos[self.m.joint(j).qposadr[0]]=v
        for name,pose in poses.items():
            jid=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_JOINT,'free:'+name)
            if jid<0: continue
            adr=self.m.jnt_qposadr[jid]; self.d.qpos[adr:adr+3]=pose['xyz']
            theta=np.deg2rad(pose.get('yaw_deg',0))/2
            self.d.qpos[adr+3:adr+7]=pose.get('quat',[np.cos(theta),0,0,np.sin(theta)])
        mujoco.mj_forward(self.m,self.d)
        self.renderer.update_scene(self.d,camera='calibrated_scene')
        frame=cv2.cvtColor(self.renderer.render(),cv2.COLOR_RGB2BGR)
        frame=cv2.remap(frame,self.mx,self.my,cv2.INTER_LINEAR)
        cv2.rectangle(frame,(0,0),(640,52),(22,22,22),-1)
        cv2.putText(frame,self.title,(12,21),cv2.FONT_HERSHEY_SIMPLEX,.53,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(frame,f'{tick/30:.1f}s | {command} | SIMULATION {self.stride/3:g}x',(12,43),cv2.FONT_HERSHEY_SIMPLEX,.45,(210,210,210),1,cv2.LINE_AA)
        self.writer.write(frame); self.frames+=1

    def close(self):
        self.writer.release(); self.renderer.close()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--trace',required=True); p.add_argument('--out',required=True)
    a=p.parse_args(); meta,ticks=read_trace(a.trace)
    outcome='SUCCESS' if meta['success'] else 'FAILURE'
    rec=Recorder(meta['scene'],meta['physics_config'],a.out,f'Expert | seed {meta["seed"]} | {outcome}')
    try:
        for i,t in enumerate(ticks): rec.frame(i,t['q'],t['poses'],t['phase'])
    finally: rec.close()
    Path(a.out).with_suffix('.json').write_text(json.dumps({'trace':a.trace,'success':meta['success'],
        'seed':meta['seed'],'frames':rec.frames,'camera':'config/camera.json','simulation':True},indent=2))
    print(a.out,rec.frames,flush=True)

if __name__=='__main__': main()
