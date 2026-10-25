local Run=game:GetService("RunService")
local deadline=os.clock()+60
while not _G.R2P_HARNESS and os.clock()<deadline do task.wait(.1) end
local H=assert(_G.R2P_HARNESS,"harness did not initialize")
local s=H.stations[1]
s:start("cube_in_bowl",1,"bot",{attract=true})
if (_G.R2P_ARGS or {}).no_collision then for _,p in ipairs(s.arm.model:GetDescendants()) do if p:IsA("BasePart") then p.CanCollide=false end end end
local start=s.arm:measured()
local frames=0; local elapsed=0; local history={}
while elapsed<2 do local dt=Run.Heartbeat:Wait(); elapsed+=dt; frames+=1; if frames<25 then table.insert(history,{t=elapsed,q=s.arm:measured()[1],sim=s.arm.sim.q[1],vel=s.arm.hinges[1].AngularVelocity}) end end
local q=s.arm:measured(); local _,_,site=s.arm:tips()
local masses={}; local velocities={}
for i,p in ipairs(s.arm.roots) do masses[i]=p.AssemblyMass end
for i,h in ipairs(s.arm.hinges) do velocities[i]=h.AngularVelocity end
local owned=true; local parts=0
for _,p in ipairs(s.model:GetDescendants()) do if p:IsA("BasePart") then parts+=1; if p:CanSetNetworkOwnership() and p:GetNetworkOwner()~=nil then owned=false end end end
return {tables=#H.stations,start_q=start,q=q,sim_q=s.arm.sim.q,command=s.commanded,site=site,owned=owned,parts=parts,frames=frames,elapsed=elapsed,active=s.active,poses=s:state().objects,masses=masses,velocities=velocities,history=history}
