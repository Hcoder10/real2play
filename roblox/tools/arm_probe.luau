local deadline=os.clock()+60
while not _G.R2P_HARNESS and os.clock()<deadline do task.wait(.1) end
local H=assert(_G.R2P_HARNESS)
local s=H.stations[1]; s:sleep(false)
for _,p in ipairs(s.arm.model:GetDescendants()) do if p:IsA("BasePart") then p.CanCollide=false end end
for _,h in ipairs(s.arm.hinges) do h.AngularVelocity=0 end
local begin=s.arm:measured()
s.arm.hinges[1].AngularVelocity=1
local start=os.clock(); local frames=0
while os.clock()-start<.6 do game:GetService("RunService").Heartbeat:Wait(); frames+=1 end
local hinge=s.arm.hinges[1]
local axis=hinge.Attachment0.WorldAxis
local velocity=s.arm.roots[2].AssemblyAngularVelocity
return {begin=begin,finish=s.arm:measured(),axis={axis.X,axis.Y,axis.Z},velocity={velocity.X,velocity.Y,velocity.Z},motor=hinge.AngularVelocity,frames=frames}
