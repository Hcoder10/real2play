local Run=game:GetService("RunService")
local Twin=game:GetService("ReplicatedStorage").Twin
local A=require(Twin.ArmBuilder)
local Scene=require(Twin.Scene)
local J=require(Twin.JointSim)
local args=_G.R2P_ARGS
local deadline=os.clock()+60
while not _G.R2P_HARNESS and os.clock()<deadline do task.wait(.1) end
local H=assert(_G.R2P_HARNESS,"harness initialization failed")
H.connection:Disconnect()
H.postConnection:Disconnect()
local s=H.stations[1]; s:sleep(false); s.arm:reset(args.q0); s.arm.sim=J.new(args.physics,args.q0)
-- First wait for engine initialization, while holding the initial configuration.
for _,h in ipairs(s.arm.hinges) do h.AngularVelocity=0 end
task.wait(.3)
s.arm:reset(args.q0)
local objects,roots=Scene.build(s.model,s.origin,args.scene)
local q,jaw={},{}
local tick=1; local elapsed=0; local acc=0
s.arm:command(args.commands[1])
local connection=Run.PreSimulation:Connect(function(dt)
    if tick>#args.commands then return end
    Scene.wakeAirborne(roots,s.origin,args.scene.objects)
    -- Roblox's simulated dt, not wall-clock/Heartbeat hitches.
    s.arm:step(dt); elapsed+=dt
end)
while tick<=#args.commands do
    Run.PostSimulation:Wait()
    if elapsed+1e-7>=tick/30 then
        table.insert(q,s.arm:measured()); local _,_,mid=s.arm:tips(); table.insert(jaw,mid)
        tick+=1; if args.commands[tick] then s.arm:command(args.commands[tick]) end
    end
end
connection:Disconnect()
return {q=q,jaw=jaw,poses=Scene.poses(roots,s.origin,args.scene.objects),simulated_s=elapsed}
