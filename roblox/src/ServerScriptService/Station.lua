local Twin=game:GetService("ReplicatedStorage").Twin
local A=require(Twin.ArmBuilder)
local K=require(Twin.Kinematics)
local M=require(Twin.Math3)
local L=require(Twin.Levels)
local C=require(Twin.TaskCheck)
local Scene=require(Twin.Scene)
local Skill=require(Twin.SkillOperator)
local Robot=require(Twin.Generated.Robot)
local Physics=require(Twin.Generated.Physics)
local Trace=require(script.Parent.Trace)
local ContactGrasp=require(script.Parent.ContactGrasp)
local TTT=require(Twin.TicTacToe)
local S={}; S.__index=S
function S.new(index,origin,parent)
    local model=Instance.new("Model"); model.Name="Table"..index; model.Parent=parent
    model:SetAttribute("Origin",origin)
    local self=setmetatable({index=index,origin=origin,model=model,owner=nil,active=false,closed=false,grabs=0,history={},input={move={0,0,0},roll=0},lastInput=0},S)
    local top=A.part(model,"Tabletop",Vector3.new(8,.35,6),CFrame.new(origin+Vector3.new(2,-.175,0)),Color3.fromRGB(225,237,246),true)
    top.CustomPhysicalProperties=PhysicalProperties.new(1,.6,0,10,10)
    local watch=Instance.new("ProximityPrompt"); watch.Name="WatchPrompt"; watch.ActionText="Watch table"; watch.ObjectText="Robot table "..index; watch.KeyboardKeyCode=Enum.KeyCode.F; watch.GamepadKeyCode=Enum.KeyCode.ButtonX; watch.MaxActivationDistance=10; watch.Parent=top; self.watchPrompt=watch
    for _,dx in ipairs({-1.5,5.5}) do for _,dz in ipairs({-2.5,2.5}) do A.part(model,"Leg",Vector3.new(.3,7.2,.3),CFrame.new(origin+Vector3.new(dx,-3.8,dz)),Color3.fromRGB(62,91,120),true) end end
    local seat=Instance.new("Seat"); seat.Name="Seat"; seat.Size=Vector3.new(2,.3,2); seat.Anchored=true
    seat.CFrame=CFrame.lookAt(origin+Vector3.new(7,-2.4,3.9),origin+Vector3.new(2,-2.4,0)); seat.Color=Color3.fromRGB(47,113,200); seat.Parent=model
    local prompt=Instance.new("ProximityPrompt"); prompt.ActionText="Sit & play"; prompt.ObjectText="Robot table "..index; prompt.HoldDuration=0; prompt.MaxActivationDistance=9; prompt.Parent=seat
    prompt.Triggered:Connect(function(player)
        if self.owner or seat.Occupant then return end
        local h=player.Character and player.Character:FindFirstChildOfClass("Humanoid")
        if h then seat:Sit(h) end
    end)
    self.seat=seat; self.prompt=prompt
    local label=Instance.new("BillboardGui"); label.Size=UDim2.fromOffset(240,72); label.StudsOffset=Vector3.new(0,2.5,0); label.MaxDistance=50; label.Parent=seat
    local text=Instance.new("TextLabel"); text.Size=UDim2.fromScale(1,1); text.BackgroundColor3=Color3.fromRGB(242,247,251); text.BackgroundTransparency=.12; text.Text="Table "..index.."  |  Sit & play"; text.TextSize=18; text.TextWrapped=true; text.Font=Enum.Font.GothamBold; text.TextColor3=Color3.fromRGB(29,52,82); text.Parent=label
    self.label=text
    self.arm=A.new(model,origin,Robot.home_q)
    self.grasp=ContactGrasp.new(self.arm)
    -- Idle tables keep the real twin but sleep its dynamic roots until occupied.
    self:sleep(true)
    local shadow=A.part(model,"GripperShadow",Vector3.new(.13,.014,.13),CFrame.new(origin+Vector3.new(2,.015,0)),Color3.fromRGB(47,113,200),true)
    shadow.Shape=Enum.PartType.Cylinder; shadow.Size=Vector3.new(.014,.16,.16); shadow.Orientation=Vector3.new(0,0,90); shadow.CanCollide=false; shadow.CanQuery=false; self.shadow=shadow
    return self
end
function S:sleep(sleeping)
    for i,root in ipairs(self.arm.roots) do if i>1 then root.Anchored=sleeping end end
    if not sleeping then A.own(self.model) end
end
function S:state()
    local _,_,site=self.arm:tips()
    return {objects=Scene.poses(self.roots or {},self.origin,self.scene and self.scene.objects or {}),gripper_open=not self.closed,site_xyz=site,initial=self.initial,q=self.arm:measured()}
end
function S:clear()
    self.active=false
    self.grasp:release()
    if self.objects then self.objects:Destroy(); self.objects=nil end
    self.roots={}; self.history={}
end
function S:start(name,seed,source,options)
    options=options or {}
    if self.trace and self.active then self:finish(false,"restarted") end
    self:clear(); self:sleep(false)
    self.task=L.copy(assert(L.tasks[name],"unknown task")); self.level=name; self.seed=seed
    if options.tictactoe then
        -- A match keeps one measured board and supply lane across every turn.
        self.task.randomize.board_pose=nil
        for id,rule in pairs(self.task.randomize or {}) do if type(rule)=="table" and id:sub(1,5)=="other" then rule.p_present=0 end end
        -- The game declares an 18 mm placement tolerance. Store that exact task
        -- variant in the trace and still judge it with the shared checker.
        for _,goal in ipairs(self.task.goal) do if goal.pred=="at" and goal.a=="piece" and goal.xy=="cell" then goal.tol_m=TTT.tolerance end end
    end
    self.scene=L.sample(self.task,seed)
    if options.boardPieces then
        local pieces=options.boardPieces
        for slot=1,math.min(4,#pieces) do
            local mark=pieces[#pieces-slot+1]
            local definition=nil
            for _,o in ipairs(self.task.objects) do if o.id=="other"..slot then definition=L.copy(o) end end
            if definition then
                local xy=TTT.xy(mark.cell); definition.xyz={xy[1],xy[2],C.size(definition)[3]/2}; definition.xy=nil
                definition.yaw_deg=0; definition.fixed=true; definition.collide=true
                definition.color=if mark.player==1 then "brown" else "blue"
                table.insert(self.scene.objects,definition)
            end
        end
    end
    self.objects,self.roots=Scene.build(self.model,self.origin,self.scene)
    self.arm:reset(Robot.home_q); self.commanded=table.clone(Robot.home_q)
    local _,_,mid=K.tips(self.commanded); self.ee=mid; self.roll=0; self.closed=false
    self.grabs=0; self.elapsed=0; self.gripHoldUntil=0; self.phaseIndex=1; self.phaseAge=0; self.commandList=L.commands(self.task)
    self.source=source or "human"; self.active=true; self.history={}; self.initial=self:state().objects
    self.limit=options.limit or self.task.limits.episode_s; self.attract=options.attract==true
    self.tictactoe=options.tictactoe; self.board=options.board; self.candidate=nil; self.hold=0
    self.operator=if self.source=="bot" then Skill.new(self.task) else nil
    self.trace=Trace.new({task=self.task,task_name=name,source=self.source,player=if self.owner then self.owner.UserId else 0,level=name,
        seed=seed,scene=L.copy(self.scene),rig=Physics.rig,physics_sha256=Physics.sha256,robot_mjcf_sha256=Robot.mjcf_sha256,
        mode=self.mode or "solo",round_limit_s=self.limit,grasp_model=if self.task.bench_calibration then "bilateral_contact_rigid" else "physical",
        tutorial=options.tutorial==true,board=options.board and table.clone(options.board)},not self.attract)
    self.trace:phase(self.commandList[1],"approach")
    self.label.Text=if self.attract then "Watch the robot" else (if self.owner then string.format("Table %d | %s\n%d points  |  Streak %d",self.index,self.owner.DisplayName,self.score or 0,self.streak or 0) else "Robot practice")
    return self
end
function S:setInput(packet)
    self.input=packet; self.lastInput=os.clock()
    if packet.grip~=nil and packet.grip~=self.closed then
        self.closed=packet.grip
        self.gripHoldUntil=self.elapsed+1.15
        if self.closed then self.grabs+=1; self.trace:phase(self.commandList[self.phaseIndex] or "home","grasp")
        else
            local cmd=self.commandList[self.phaseIndex] or "release"
            if cmd:sub(1,5)~="place" and cmd:sub(1,7)~="push_to" then cmd="release" end
            self.trace:phase(cmd,"release")
        end
        self.phaseAge=0
        if self.onEvent then self.onEvent(self,if self.closed then "grab" else "drop") end
    end
end
function S:phaseUpdate(state)
    self.phaseAge=(self.phaseAge or 0)+1/30
    local step=self.task.skeleton[self.phaseIndex]
    if not step then return end
    local advance=false
    if step.skill=="pick" then advance=self.closed and self.phaseAge>.35 and state.site_xyz[3]>.07
    elseif step.skill:sub(1,5)=="place" or step.skill=="push_to" then advance=not self.closed and self.grabs>0 and self.phaseAge>.2
    elseif step.skill=="release" then advance=not self.closed and self.phaseAge>.5
    elseif step.skill=="wait" then advance=self.phaseAge>=step.s end
    if advance then
        self.phaseIndex=math.min(#self.commandList,self.phaseIndex+1); self.phaseAge=0
        self.trace:phase(self.commandList[self.phaseIndex],self.task.skeleton[self.phaseIndex].skill)
    end
end
function S:patchCell(cell)
    local xy=TTT.xy(cell); local p=self.roots.cell
    local dimensions; local yaw=0
    for _,o in ipairs(self.task.objects) do if o.id=="cell" then dimensions=o.size_m; yaw=o.yaw_deg or 0 end end
    if p then p.CFrame=CFrame.new(self.origin+A.vector({xy[1],xy[2],.0005}))*CFrame.Angles(0,math.rad(yaw),0) end
    for _,o in ipairs(self.scene.objects) do if o.id=="cell" then o.xyz={xy[1],xy[2],.0005} end end
    for _,o in ipairs(self.trace.meta.scene.objects) do if o.id=="cell" then o.xyz={xy[1],xy[2],.0005} end end
    -- The chosen target is late-bound; every training slot must contain that target.
    for _,tick in ipairs(self.trace.ticks) do
        tick.poses.cell={xyz={xy[1],xy[2],.0005},yaw_deg=yaw,size_m=dimensions}
        tick.slots=L.slots(self.task,tick.poses)
    end
end
function S:tick()
    if not self.active then return end
    Scene.wakeAirborne(self.roots,self.origin,self.scene.objects)
    local state=self:state(); local measured=self.arm:measured()
    if self.operator then
        local target,command,phase=self.operator:step(state,1/30)
        self.commanded=target; self.closed=if self.operator.gripClosed~=nil then self.operator.gripClosed else target[6]<.15; self.trace:phase(command,phase)
    else
        local input=self.input
        if os.clock()-self.lastInput>.35 then input={move={0,0,0},roll=0} end
        local speed=if input.fine then .025 else .085
        self.ee={math.clamp(self.ee[1]+input.move[1]*speed/30,.10,.34),math.clamp(self.ee[2]+input.move[2]*speed/30,-.24,.24),math.clamp(self.ee[3]+input.move[3]*speed/30,.007,.23)}
        self.roll=math.clamp(self.roll+input.roll*.9/30,-2.7,2.8)
        local q=table.clone(self.commanded)
        for i=1,6 do if math.abs(q[i]-measured[i])>math.rad(4) then q=measured; break end end
        q[6]=if self.closed then -.16 else .95
        local still=input.move[1]==0 and input.move[2]==0 and input.move[3]==0 and input.roll==0
        if self.elapsed<self.gripHoldUntil and still then
            -- A grip-only command moves the jaw, not the fixed finger. Rebase
            -- the next Cartesian command on the naturally shifted midpoint.
            self.commanded=table.clone(self.commanded); self.commanded[6]=q[6]
            self.ee=table.clone(state.site_xyz)
        else
            q=K.ik(self.ee,q,self.roll,5); self.commanded=Skill.compensate(q)
        end
        self:phaseUpdate(state)
    end
    if self.task.bench_calibration then
        local jawFloor=self.grasp:update(self.closed,self.roots,self.scene.objects)
        if jawFloor then self.commanded[6]=math.max(jawFloor,self.commanded[6]) end
    end
    self.arm:command(self.commanded)
    self.trace:tick(measured,self.commanded,state.objects,self.closed,L.slots(self.task,state.objects))
    self.elapsed+=1/30
    table.insert(self.history,state); if #self.history>15 then table.remove(self.history,1) end
    if self.tictactoe then
        local piece=state.objects.piece
        local cell=piece and TTT.cell(piece.xyz,self.board)
        local atRest=piece and piece.v<.015 and piece.xyz[3]<.025 and not self.closed
        if cell and atRest then
            if cell==self.candidate then self.hold+=1/30 else self.candidate=cell; self.hold=1/30 end
        else self.candidate=nil; self.hold=0 end
        if self.hold>=.5-1e-8 then
            self:patchCell(cell)
            -- Re-evaluate the canonical checker for the late-bound target.
            for _,s in ipairs(self.history) do s.objects.cell={xyz={TTT.xy(cell)[1],TTT.xy(cell)[2],.0005},yaw_deg=0} end
            if C.check_window(self.task,self.history).ok then self:finish(true,"",{target_cell=cell}); return end
        end
    elseif C.check_window(self.task,self.history).ok then self:finish(true,""); return end
    for _,p in pairs(state.objects) do if p.xyz[3]<-.12 then self:finish(false,"fell_off_table"); return end end
    if self.elapsed>=self.limit then self:finish(false,"timeout") end
end
function S:step(dt)
    if self.active then self.arm:step(dt) end
    local _,_,site=self.arm:tips()
    self.shadow.Position=self.origin+A.vector({site[1],site[2],.002})
end
function S:finish(success,reason,extra)
    if not self.active then return end
    self.active=false; self.trace:finish(success,reason,extra)
    local trace=self.trace; local owner=self.owner
    trace:upload(function(ok,detail) if self.onUpload then self.onUpload(self,ok,detail,owner) end end)
    if self.onFinish then self.onFinish(self,success,reason,extra or {}) end
end
return S
