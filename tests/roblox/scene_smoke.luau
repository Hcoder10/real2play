local deadline=os.clock()+60
while not _G.R2P_HARNESS and os.clock()<deadline do task.wait(.1) end
assert(_G.R2P_HARNESS,"harness unavailable")
local Twin=game:GetService("ReplicatedStorage").Twin
local Scene=require(Twin.Scene)
local origin=Vector3.new(0,20,0)
local definitions={
    {id="ball",shape="sphere",size_m={.04},xyz={.1,0,.1},yaw_deg=25,mass_kg=.04},
    {id="upright",shape="cylinder",size_m={.04,.06},axis="z",xyz={.2,0,.1},yaw_deg=-37,mass_kg=.04},
    {id="along_y",shape="cylinder",size_m={.04,.06},axis="y",xyz={.3,0,.1},yaw_deg=140,mass_kg=.04},
    {id="bowl",shape="bowl",size_m={.12,.05},xyz={.4,0,.1},yaw_deg=17,mass_kg=.25},
}
local model,roots=Scene.build(workspace,origin,{objects=definitions})
local poses=Scene.poses(roots,origin,definitions); local checks=0
local before={}
for id,p in pairs(roots) do before[id]={touch=p.CanTouch,collide=p.CanCollide,cf=p.CFrame,v=p.AssemblyLinearVelocity,w=p.AssemblyAngularVelocity} end
Scene.wakeAirborne(roots,origin,definitions)
for _,o in ipairs(definitions) do
    local p=roots[o.id]; local pose=poses[o.id]
    assert(math.abs(p.AssemblyMass-o.mass_kg)<.0001,"incorrect assembly mass: "..o.id)
    for axis=1,3 do assert(math.abs(pose.xyz[axis]-o.xyz[axis])<1e-6,"incorrect object centre") end
    assert(math.abs((pose.yaw_deg-o.yaw_deg+180)%360-180)<.001,"cylinder geometry rotation leaked into canonical pose")
    assert(p:GetNetworkOwner()==nil,"object is not server owned")
    local prior=before[o.id]
    assert(p.CanTouch~=prior.touch,"airborne assembly was not awakened")
    assert(p.CanCollide==prior.collide and p.CFrame==prior.cf,"wake changed collision or pose")
    assert(p.AssemblyLinearVelocity==prior.v and p.AssemblyAngularVelocity==prior.w,"wake injected velocity")
    checks+=9
end
model:Destroy()
return {pass=true,shape_pose_mass_checks=checks}
