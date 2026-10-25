local Players=game:GetService("Players")
local Run=game:GetService("RunService")
local Config=require(game:GetService("ServerStorage").TwinConfig)
Config.TRACE_URL=_G.R2P_ARGS.trace_url; Config.TRACE_KEY=_G.R2P_ARGS.trace_key; Config.BOT_IDLE=false
local deadline=os.clock()+60
while (not _G.R2P_HARNESS or #Players:GetPlayers()==0) and os.clock()<deadline do task.wait(.1) end
local H=assert(_G.R2P_HARNESS,"harness unavailable")
assert(#H.stations==20,"expected twenty tables")
local player=assert(Players:GetPlayers()[1],"Play mode must create a real Player")
local character=player.Character or player.CharacterAdded:Wait()
deadline=os.clock()+15
while not character:IsDescendantOf(workspace) and os.clock()<deadline do task.wait(.1) end
assert(character:IsDescendantOf(workspace),"player character never entered Workspace")
local humanoid=character:WaitForChild("Humanoid")
local station=H.stations[1]
local root=character:WaitForChild("HumanoidRootPart")
root:SetNetworkOwner(nil)
character:PivotTo(station.seat.CFrame*CFrame.new(0,2,0))
task.wait(.5)
station.seat:Sit(humanoid)
deadline=os.clock()+10
while H.byPlayer[player]~=station and os.clock()<deadline do station.seat:Sit(humanoid); task.wait(.3) end
assert(station.seat.Occupant==humanoid and H.byPlayer[player]==station,"seat ownership did not start; occupant="..tostring(station.seat.Occupant).." health="..humanoid.Health.." owner="..tostring(station.owner).." seat="..tostring(humanoid.SeatPart))
local inspect=Instance.new("RemoteFunction"); inspect.Name="R2PSmokeInspect"; inspect.Parent=game:GetService("ReplicatedStorage")
task.wait(.4)
local Rules=require(game:GetService("ReplicatedStorage").Twin.TicTacToe)
local A=require(game:GetService("ReplicatedStorage").Twin.ArmBuilder)
local moves=0; local traces={}
assert(H.begin(player,"place_on_cell","solo"),"judge's default placement level failed to start")
local skillTrace=station.trace; skillTrace.meta.source="bot"; skillTrace.meta.test_only=true; skillTrace.meta.test_fixture="highlighted_cap_placement"
table.insert(traces,skillTrace.id)
local target=station:state().objects.cell.xyz
local cap=station.roots.piece
cap.CFrame=CFrame.new(station.origin+A.vector({target[1],target[2],Rules.pieceSize[3]/2}))*(cap:GetAttribute("ObjectRotation") or CFrame.identity):Inverse()
cap.AssemblyLinearVelocity=Vector3.zero; cap.AssemblyAngularVelocity=Vector3.zero
deadline=os.clock()+8
while not skillTrace.finished and os.clock()<deadline do task.wait(.05) end
assert(skillTrace.meta.success,"highlighted cap fixture did not satisfy the shared checker")
assert(station.score>0 and station.streak==1 and station.model:GetAttribute("Streak")==1,"per-table score/streak missing")
local earned=station.score
task.wait(.3)
assert(H.begin(player,"place_on_cell","tic_tac_toe"),"match failed to start")
local client=inspect:InvokeClient(player)
assert(client.gui and client.locked and client.controls,"phone controls or seated camera missing")
assert(station.score==earned and station.label.Text:find("Streak 1"),"next round lost the per-table score")
assert(not H.watch(player,H.stations[2]),"seated player must retain their table instead of spectating")
-- Fixture placements exercise the real server success window and full match loop.
-- They are explicitly bot/test-only traces, never presented as physical teleop success.
while station.match and station.match.alive and moves<5 do
    deadline=os.clock()+6
    while not station.active and station.match.alive and os.clock()<deadline do task.wait(.05) end
    if not station.match.alive then break end
    assert(station.active,"next player turn did not start")
    local trace=station.trace; trace.meta.source="bot"; trace.meta.test_only=true; trace.meta.test_fixture="tic_tac_toe_placement"
    table.insert(traces,trace.id)
    local cell=Rules.legal(station.match.board)[1]; assert(cell,"no legal human move")
    local xy=Rules.xy(cell); local piece=station.roots.piece
    station:setInput({move={0,0,0},roll=0,grip=false})
    -- A 16 mm offset must count for the game's declared 18 mm tolerance, even
    -- though the standalone placement skill asks for a tighter 12 mm target.
    local offset=.016/math.sqrt(2)
    piece.CFrame=CFrame.new(station.origin+A.vector({xy[1]+offset,xy[2]+offset,Rules.pieceSize[3]/2}))*(piece:GetAttribute("ObjectRotation") or CFrame.identity):Inverse(); piece.AssemblyLinearVelocity=Vector3.zero; piece.AssemblyAngularVelocity=Vector3.zero
    deadline=os.clock()+8
    while station.trace==trace and not trace.finished and os.clock()<deadline do task.wait(.05) end
    assert(trace.finished and trace.meta.success,"placement hold did not finish: "..tostring(trace.meta.fail_reason))
    moves+=1
    task.wait(1)
end
assert(station.match and not station.match.alive,"tic-tac-toe match did not finish")
assert(Rules.result(station.match.board)~=nil,"board not terminal")
local owned=true
local offenders={}
for _,p in ipairs(H.world:GetDescendants()) do if p:IsA("BasePart") and p:CanSetNetworkOwnership() and (p:GetNetworkOwner()~=nil or p:GetNetworkOwnershipAuto()) then owned=false; table.insert(offenders,p:GetFullName()) end end
assert(owned,"twin ownership became automatic or client-owned: "..table.concat(offenders,", "))
local outcome=Rules.result(station.match.board)
humanoid.Sit=false
deadline=os.clock()+5
while H.byPlayer[player] and os.clock()<deadline do task.wait(.1) end
assert(not station.owner and not H.byPlayer[player],"standing did not free table")
task.wait(1)
local lobby=inspect:InvokeClient(player)
assert(lobby.camera_restored,"standing did not restore the player camera")
assert(H.watch(player,H.stations[2]),"spectator view refused")
task.wait(.5)
local spectator=inspect:InvokeClient(player)
assert(spectator.locked and not spectator.controls and not H.byPlayer[player],"spectator controls or camera incorrect")
assert(spectator.table_score:find("Table 2"),"spectator table score missing")
inspect:InvokeClient(player,"stand")
task.wait(.5)
assert(not H.watchers[player] and inspect:InvokeClient(player).camera_restored,"leaving spectator view failed")
return {pass=true,tables=#H.stations,seated=true,client=client,result=outcome,moves=moves,trace_ids=traces,server_owned=owned,stood_up=true,spectator=true,fixture="bot/test_only placements; manipulation covered separately"}
