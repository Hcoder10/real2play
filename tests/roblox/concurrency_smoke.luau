local Config=require(game:GetService("ServerStorage").TwinConfig)
Config.BOT_IDLE=false
local deadline=os.clock()+60
while not _G.R2P_HARNESS and os.clock()<deadline do task.wait(.1) end
local H=assert(_G.R2P_HARNESS)
local L=require(game:GetService("ReplicatedStorage").Twin.Levels)
for i=1,8 do H.stations[i]:start(L.order[(i-1)%#L.order+1],100+i,"bot",{attract=true,limit=20}) end
local frames=0; local start=os.clock(); local elapsed=0
while elapsed<5 do local dt=game:GetService("RunService").Heartbeat:Wait(); elapsed+=dt; frames+=1 end
local owned=true; local ticks={}
for i=1,8 do
    local s=H.stations[i]; ticks[i]=math.floor(s.elapsed*30+.5)
    assert(ticks[i]>30,"station did not advance independently")
    for _,p in ipairs(s.model:GetDescendants()) do if p:IsA("BasePart") and p:CanSetNetworkOwnership() and p:GetNetworkOwner()~=nil then owned=false end end
end
assert(owned,"client ownership detected")
return {pass=true,active_tables=8,server_owned=owned,control_ticks=ticks,frames=frames,wall_s=os.clock()-start,heartbeat_s=elapsed}
