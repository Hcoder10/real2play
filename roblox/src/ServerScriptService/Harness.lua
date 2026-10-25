local Players=game:GetService("Players")
local Run=game:GetService("RunService")
local PhysicsService=game:GetService("PhysicsService")
local Twin=game:GetService("ReplicatedStorage").Twin
local Station=require(script.Parent.Station)
local Match=require(script.Parent.TicTacToeMatch)
local Progress=require(script.Parent.Progress)
local Trace=require(script.Parent.Trace)
local Touch=require(Twin.TouchMap)
local Score=require(Twin.Score)
local Levels=require(Twin.Levels)
local A=require(Twin.ArmBuilder)
local G=require(Twin.Generated.Game)
local Physics=require(Twin.Generated.Physics)
local Config=require(game:GetService("ServerStorage").TwinConfig)
local H={stations={},byPlayer={},watchers={},started=false}
local function send(player,data) if player and player.Parent then H.state:FireClient(player,data) end end
function H.snapshot(station,kind,extra)
    local player=station.owner; if not player then return end
    local stats=Progress.get(player)
    local data={kind=kind,table=station.index,origin={station.origin.X,station.origin.Y,station.origin.Z},level=station.level,mode=station.mode,
        remaining=math.max(0,(station.limit or 0)-(station.elapsed or 0)),stats=stats,board=station.match and station.match.board,
        image=Config.REAL_ARM_IMAGE,unlocks=Score.unlocks(stats.stars),leaders=Progress.sessionLeaders(),table_score=station.score or 0,table_streak=station.streak or 0}
    for k,v in pairs(extra or {}) do data[k]=v end
    send(player,data)
end
function H.tableScore(station,reward,success)
    station.score=(station.score or 0)+(reward.points or 0)
    station.streak=if success then (station.streak or 0)+1 else 0
    station.model:SetAttribute("Score",station.score); station.model:SetAttribute("Streak",station.streak)
    station.label.Text=string.format("Table %d | %s\n%d points  |  Streak %d",station.index,station.owner.DisplayName,station.score,station.streak)
end
function H.watch(player,station)
    if H.byPlayer[player] then return false end
    H.watchers[player]=station
    local origin=station.origin
    send(player,{kind="spectate",table=station.index,origin={origin.X,origin.Y,origin.Z},table_score=station.score or 0,table_streak=station.streak or 0})
    return true
end
function H.release(player)
    local station=H.byPlayer[player]; if not station then return end
    if station.match then station.match:destroy(); station.match=nil end
    if station.active then station:finish(false,"stood_up") end
    station.owner=nil; H.byPlayer[player]=nil; station.prompt.Enabled=true; station.score=0; station.streak=0; station.label.Text="Table "..station.index.." | Sit & play"
    station.model:SetAttribute("Score",0); station.model:SetAttribute("Streak",0); station.model:SetAttribute("Occupied",false)
    station:clear(); station:sleep(true); station:display(); send(player,{kind="lobby",stats=Progress.get(player),image=Config.REAL_ARM_IMAGE,preview=false})
end
function H.sit(player,station)
    if station.owner and station.owner~=player then return false end
    if H.byPlayer[player] and H.byPlayer[player]~=station then H.release(player) end
    if station.active then station:finish(false,"player_joined") end
    H.watchers[player]=nil
    station.owner=player; station.prompt.Enabled=false; H.byPlayer[player]=station; station.model:SetAttribute("Occupied",true)
    H.tableScore(station,{points=0},false)
    H.snapshot(station,"seated",{levels=Levels.order,tutorial=G.tutorial})
    return true
end
function H.begin(player,level,mode)
    local station=H.byPlayer[player]; if not station then return false end
    if not table.find(G.modes,mode) then return false end
    if station.match then station.match:destroy(); station.match=nil end
    if station.active then station:finish(false,"restarted") end
    station.mode=mode; station.opponent=nil
    if mode=="tic_tac_toe" then
        station.match=Match.new(station,function(match,result,reason)
            local reward=Progress.record(player,"tic_tac_toe",result~=2,station.elapsed or 0,G.move_s,station.grabs,mode,result)
            H.tableScore(station,reward,result~=2)
            H.snapshot(station,"result",{success=result~=2,result=result,reward=reward,reason=reason})
        end,function(_,phase) H.snapshot(station,"board",{phase=phase}) end)
        station.match:startMove()
    else
        if mode=="daily" then level=Levels.order[Score.daily(os.date("!%Y-%m-%d"))%#Levels.order+1] end
        if not Levels.tasks[level] then return false end
        local seed=if mode=="daily" then Score.daily(os.date("!%Y-%m-%d")) else math.random(1,2147483646)
        if mode=="solo" and level==G.default_level and Progress.get(player).rounds==0 and G.intro_seed then seed=G.intro_seed end
        station:start(level,seed,"human",{limit=if mode=="timed" then G.round_s else nil,tutorial=Progress.get(player).rounds==0})
        -- vs bot races a visible adjacent idle table on the identical seed.
        if mode=="vs_bot" then
            for _,opponent in ipairs(H.stations) do if opponent~=station and not opponent.owner and not opponent.active then
                opponent:start(level,seed,"bot"); station.opponent=opponent; break
            end end
        end
    end
    H.snapshot(station,"start",{instruction=station.task.instruction,tutorial=if Progress.get(player).rounds==0 then G.tutorial else nil})
    return true
end
local function isOwner(player)
    return (game.CreatorType==Enum.CreatorType.User and player.UserId==game.CreatorId) or table.find(Config.OWNER_IDS,player.UserId)~=nil or (Run:IsStudio() and player.UserId<=0)
end
local function onPlayer(player)
    task.spawn(Progress.load,player)
    player.Chatted:Connect(function(message)
        if not isOwner(player) then return end
        local url=message:match("^/traceurl%s+(%S+)%s*$")
        if url and (url:match("^https://") or (Run:IsStudio() and url:match("^http://localhost:"))) then Config.TRACE_URL=url:gsub("/$",""); send(player,{kind="notice",text="Trace destination updated."}) end
        local image=message:match("^/realarm%s+(%d+)%s*$")
        if image then Config.REAL_ARM_IMAGE="rbxassetid://"..image; H.state:FireAllClients({kind="image",image=Config.REAL_ARM_IMAGE}) end
    end)
    local function character(char)
        for _,part in ipairs(char:GetDescendants()) do if part:IsA("BasePart") then part.CollisionGroup="R2PPlayer" end end
        char.DescendantAdded:Connect(function(part) if part:IsA("BasePart") then part.CollisionGroup="R2PPlayer" end end)
    end
    player.CharacterAdded:Connect(character); if player.Character then character(player.Character) end
end
function H.start()
    if H.started then return H end; H.started=true
    for _,name in ipairs({"R2PArm","R2PObject","R2PPlayer"}) do pcall(function() PhysicsService:RegisterCollisionGroup(name) end) end
    PhysicsService:CollisionGroupSetCollidable("R2PArm","R2PArm",false)
    PhysicsService:CollisionGroupSetCollidable("R2PArm","R2PPlayer",false)
    PhysicsService:CollisionGroupSetCollidable("R2PObject","R2PPlayer",false)
    workspace.Gravity=-Physics.config.gravity[3]*10
    local world=Instance.new("Model"); world.Name="Real2Play"; world.Parent=workspace; H.world=world
    A.part(world,"WorkshopFloor",Vector3.new(105,.5,88),CFrame.new(25,-.25,24),Color3.fromRGB(35,74,82),true)
    A.part(world,"BackWall",Vector3.new(105,18,.4),CFrame.new(25,9,-19),Color3.fromRGB(229,239,247),true)
    local sign=A.part(world,"WorkshopSign",Vector3.new(31,5,.2),CFrame.new(25,12,-18.7),Color3.fromRGB(47,113,200),true)
    local face=Instance.new("SurfaceGui"); face.Face=Enum.NormalId.Back; face.CanvasSize=Vector2.new(1000,160); face.Parent=sign
    local title=Instance.new("TextLabel"); title.Size=UDim2.fromScale(1,1); title.BackgroundTransparency=1; title.Text="Teach the robot"; title.TextScaled=true; title.Font=Enum.Font.FredokaOne; title.TextColor3=Color3.fromRGB(242,206,92); title.Parent=face
    local spawn=Instance.new("SpawnLocation"); spawn.Name="Welcome"; spawn.Size=Vector3.new(5,.2,5); spawn.CFrame=CFrame.new(25,.1,-11); spawn.Anchored=true; spawn.Transparency=1; spawn.Neutral=true; spawn.Parent=world
    H.input=Instance.new("RemoteEvent"); H.input.Name="Input"; H.input.Parent=Twin
    H.state=Instance.new("RemoteEvent"); H.state.Name="State"; H.state.Parent=Twin
    H.action=Instance.new("RemoteEvent"); H.action.Name="Action"; H.action.Parent=Twin
    for i=1,G.tables do
        local origin=Vector3.new(((i-1)%5)*18-13,7.5,math.floor((i-1)/5)*19)
        local station=Station.new(i,origin,world); H.stations[i]=station
        station.watchPrompt.Triggered:Connect(function(player)
            H.watch(player,station)
        end)
        station.seat:GetPropertyChangedSignal("Occupant"):Connect(function()
            local humanoid=station.seat.Occupant
            local player=humanoid and Players:GetPlayerFromCharacter(humanoid.Parent)
            if player then H.sit(player,station) elseif station.owner then H.release(station.owner) end
        end)
        station.onFinish=function(s,success,reason,extra)
            if s.attract then return end
            if s.owner then
                if s.match and s.match.alive then s.match:accepted(success,reason,extra)
                else
                    local reward=Progress.record(s.owner,s.level,success,s.elapsed,s.limit,s.grabs,s.mode)
                    H.tableScore(s,reward,success)
                    local beatBot=if s.opponent then s.opponent.active or not s.opponent.trace.meta.success else nil
                    H.snapshot(s,"result",{success=success,reason=reason,reward=reward,beat_bot=beatBot})
                end
            end
        end
        station.onUpload=function(s,ok,detail,owner)
            local player=owner -- credit the captured round owner, even if the seat changed
            if player then
                if ok then Progress.get(player).sent+=1; task.spawn(Progress.save,player) end
                send(player,{kind="upload",stats=Progress.get(player),uploaded=ok,upload_reason=if ok then "" else detail})
            end
        end
        station.onEvent=function(s,event) H.snapshot(s,"juice",{effect=event}) end
    end
    H.input.OnServerEvent:Connect(function(player,packet)
        local station=H.byPlayer[player]
        if not station or not station.active or station.source~="human" then return end
        local now=os.clock(); if now-(station.lastPacket or 0)<1/70 then return end; station.lastPacket=now
        local clean=Touch.sanitize(packet); if clean then station:setInput(clean) end
    end)
    local actions={}
    H.action.OnServerEvent:Connect(function(player,action,args)
        local now=os.clock(); if now-(actions[player] or 0)<.25 then return end; actions[player]=now
        if action=="ready" then send(player,{kind="lobby",stats=Progress.get(player),image=Config.REAL_ARM_IMAGE,levels=Levels.order,leaders=Progress.sessionLeaders()})
        elseif action=="scores" then send(player,{kind="scores",leaders=Progress.sessionLeaders()})
        elseif action=="play" and type(args)=="table" then H.begin(player,args.level,args.mode)
        elseif action=="stand" then
            if H.watchers[player] then H.watchers[player]=nil; send(player,{kind="lobby",stats=Progress.get(player),preview=false}) end
            local s=H.byPlayer[player]; if s and s.seat.Occupant then s.seat.Occupant.Sit=false end; H.release(player)
        elseif action=="color" then local s=H.byPlayer[player]; if s and type(args)=="number" then local unlock=G.unlocks[args]; if unlock and Progress.get(player).stars>=unlock.stars then
            for _,part in ipairs(s.arm.model:GetChildren()) do
                local body=part:GetAttribute("Body")
                if part:IsA("MeshPart") and (body=="gripper" or body=="moving_jaw_so101_v1") and part:GetAttribute("Paintable") then part.Color=Color3.fromRGB(unpack(unlock.color)) end
            end
        end end end
    end)
    Players.PlayerAdded:Connect(onPlayer); for _,player in ipairs(Players:GetPlayers()) do onPlayer(player) end
    Players.PlayerRemoving:Connect(function(player) H.release(player); H.watchers[player]=nil; Progress.save(player); actions[player]=nil end)
    local acc=0; local ui=0; local idle=0; local botIndex=0
    H.connection=Run.PreSimulation:Connect(function(dt)
        -- Advance with simulated physics time; wall-clock Studio stalls are not
        -- extra simulation ticks. This is the same schedule as the parity runner.
        acc+=dt; ui+=dt; idle+=dt
        for _,s in ipairs(H.stations) do if s.active then s:step(dt) end end
    end)
    H.postConnection=Run.PostSimulation:Connect(function()
        while acc>=1/30 do acc-=1/30; for _,s in ipairs(H.stations) do if s.active then s:tick() end end end
        if ui>.2 then
            ui=0; for _,s in ipairs(H.stations) do if s.owner and s.active then H.snapshot(s,"tick") end end
            for player,s in pairs(H.watchers) do send(player,{kind="watch_tick",table=s.index,table_score=s.score or 0,table_streak=s.streak or 0,teacher=s.owner and s.owner.DisplayName or "Open table"}) end
        end
        if idle>Config.BOT_INTERVAL then
            idle=0
            local attract=H.stations[#H.stations]
            if not workspace:GetAttribute("R2PTest") and not attract.owner and not attract.active then attract:start("bowl_bench",7,"bot",{attract=true}) end
            if Config.BOT_IDLE and #Players:GetPlayers()==0 then
                local s=H.stations[1]
                if not s.active then botIndex+=1; s:start(Levels.order[(botIndex-1)%#Levels.order+1],botIndex,"bot") end
            end
        end
    end)
    -- A separate unrecorded demonstration remains behind the lobby menu.
    if not workspace:GetAttribute("R2PTest") then H.stations[#H.stations]:start("bowl_bench",7,"bot",{attract=true}) end
    task.spawn(function() while H.started do Progress.leaders(); task.wait(120) end end)
    game:BindToClose(function()
        for _,player in ipairs(Players:GetPlayers()) do H.release(player); task.spawn(Progress.save,player) end
        local deadline=os.clock()+25
        while Trace.pending>0 and os.clock()<deadline do task.wait(.1) end
    end)
    _G.R2P_HARNESS=H
    return H
end
return H
