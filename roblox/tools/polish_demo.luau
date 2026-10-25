-- Studio-only recorded demonstration. Real physics, bot/test-only traces.
local Players=game:GetService("Players")
local Config=require(game.ServerStorage.TwinConfig)
Config.BOT_IDLE=false; Config.TRACE_URL=""; Config.TRACE_KEY=""
while not _G.R2P_HARNESS or #Players:GetPlayers()==0 do task.wait(.1) end
local H=_G.R2P_HARNESS
local player=Players:GetPlayers()[1]
local character=player.Character or player.CharacterAdded:Wait()
local humanoid=character:WaitForChild("Humanoid")
local root=character:WaitForChild("HumanoidRootPart")
local stage=Instance.new("StringValue"); stage.Name="DemoStage"; stage.Parent=game.ReplicatedStorage
local inspect=Instance.new("RemoteFunction"); inspect.Name="PolishInspect"; inspect.Parent=game.ReplicatedStorage
local report=Instance.new("StringValue"); report.Name="DemoReport"; report.Parent=game.ReplicatedStorage
local auditAction=Instance.new("StringValue"); auditAction.Name="DemoAuditAction"; auditAction.Parent=game.ReplicatedStorage
auditAction.Changed:Connect(function(action)
    if action=="walk" or action=="occupied" then
        humanoid.Sit=false; H.release(player); task.wait(.5)
        character:PivotTo(H.stations[2].seat.CFrame*CFrame.new(0,2,2))
        H.stations[2].model:SetAttribute("Occupied",action=="occupied")
        stage.Value="audit_"..action
    else inspect:InvokeClient(player,action) end
end)
local samples={}
task.spawn(function()
    while true do
        task.wait(2)
        local ok,result=pcall(function() return inspect:InvokeClient(player) end)
        if ok then result.stage=stage.Value; result.cap_success=stage:GetAttribute("CapSuccess"); result.cap_sim_seconds=stage:GetAttribute("CapSimSeconds"); table.insert(samples,result); report.Value=game.HttpService:JSONEncode(samples) end
    end
end)
task.wait(5)
stage.Value="lobby"
task.wait(24)
local station=H.stations[1]
root:SetNetworkOwner(nil); character:PivotTo(station.seat.CFrame*CFrame.new(0,2,0))
task.wait(.5); station.seat:Sit(humanoid)
local deadline=os.clock()+8
while H.byPlayer[player]~=station and os.clock()<deadline do station.seat:Sit(humanoid); task.wait(.2) end
assert(H.byPlayer[player]==station,"demo player did not sit")
stage.Value="seated"
task.wait(3)
assert(H.begin(player,"place_on_cell","solo"))
station.operator=require(game.ReplicatedStorage.Twin.SkillOperator).new(station.task)
station.source="bot"; station.trace.meta.source="bot"; station.trace.meta.test_only=true
station.trace.meta.test_fixture="Y2 recorded physical bot demo"
stage.Value="cap"
deadline=os.clock()+65
while station.active and os.clock()<deadline do task.wait(.1) end
stage.Value="cap_result_"..tostring(station.trace.meta.success)
stage:SetAttribute("CapSuccess",station.trace.meta.success)
stage:SetAttribute("CapSimSeconds",station.elapsed)
task.wait(5)
assert(H.begin(player,"bowl_bench","solo"))
station.operator=require(game.ReplicatedStorage.Twin.SkillOperator).new(station.task)
station.source="bot"; station.trace.meta.source="bot"; station.trace.meta.test_only=true
station.trace.meta.test_fixture="Y2 recorded physical bowl bot demo"
stage.Value="bowl"
task.wait(25)
stage.Value="done"
while true do task.wait(1) end
