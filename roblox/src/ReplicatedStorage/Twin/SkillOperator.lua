-- Task-skeleton operator; emits the same phase events as human control.
local K=require(if script then script.Parent.Kinematics else "./Kinematics")
local M=require(if script then script.Parent.Math3 else "./Math3")
local C=require(if script then script.Parent.TaskCheck else "./TaskCheck")
local L=require(if script then script.Parent.Levels else "./Levels")
local J=require(if script then script.Parent.JointSim else "./JointSim")
local Robot=require(if script then script.Parent.Generated.Robot else "./Generated/Robot")
local Physics=require(if script then script.Parent.Generated.Physics else "./Generated/Physics")
local S={}; S.__index=S
function S.new(task,q)
    return setmetatable({task=task,index=1,stage=1,age=0,total=0,q=table.clone(q or Robot.home_q),commands=L.commands(task),done=false},S)
end
function S.compensate(q)
    local g=J.inverse(q,table.create(6,0),table.create(6,0),-Physics.config.gravity[3]); local out=table.clone(q)
    for i,name in ipairs(Robot.joints) do
        local kp=Physics.config.joints[name].kp
        out[i]=math.clamp(q[i]+g[i]/kp,Robot.range[i][1],Robot.range[i][2])
    end
    return out
end
function S:advance()
    self.stage+=1; self.age=0
end
function S:next()
    self.index+=1; self.stage=1; self.age=0; self.retry=0; self.pickYaw=nil; self.placeOffset=nil
    if self.index>#self.task.skeleton then self.done=true end
end
function S:step(state,dt)
    if self.done then return S.compensate(self.q),"home","done",true end
    self.age+=dt; self.total+=dt
    local step=self.task.skeleton[self.index]; local command=self.commands[self.index]
    local poses=state.objects; local target=nil; local grip=self.q[6]
    local def=nil; for _,o in ipairs(self.task.objects) do if o.id==step.obj then def=o end end
    local p=step.obj and poses[step.obj]
    local skill=step.skill; local phase=skill..":"..self.stage
    if (skill=="pick" or skill:sub(1,5)=="place" or skill=="push_to") and not p then self:next(); return self.q,command,"absent",false end
    if skill=="pick" then
        local size=C.size(def,p.size_m); local xyz=p.xyz
        if not self.pickYaw then
            local bearing=math.atan2(xyz[2],xyz[1]); local yaw=math.rad(p.yaw_deg or 0)+(if size[1]>size[2]+1e-6 then math.pi/2 else 0)
            local period=if math.abs(size[1]-size[2])<1e-6 then math.pi/2 else math.pi
            self.pickYaw=bearing+(yaw-bearing+period/2)%period-period/2; self.yaw=self.pickYaw
        end
        local clearance=if def.shape=="box" and size[3]<.02 then .0015 else .0035
        local z=math.max(clearance,xyz[3]-size[3]/2+clearance)
        local thin=def.shape=="box" and size[3]<.02
        target={xyz[1],xyz[2],if self.stage==1 then (if thin then .09 else .10) elseif self.stage==4 then (if thin then .07 else .10) else z}
        grip=if self.stage<3 then math.clamp((math.min(size[1],size[2])+.018-.017)/.081,.35,.95) else -.16
        if self.stage==1 and (M.norm(M.sub(state.site_xyz,target))<.008 or self.age>6) then self:advance()
        elseif self.stage==2 and (M.norm(M.sub(state.site_xyz,target))<.005 or self.age>4) then self:advance()
        elseif self.stage==3 and self.age>(if thin then 1.15 else .85) then self:advance()
        elseif self.stage==4 and self.age>1.2 and xyz[3]-size[3]/2>.04 then self:next()
        elseif self.stage==4 and self.age>3.5 then
            if (self.retry or 0)<2 then self.retry=(self.retry or 0)+1; self.stage=1; self.age=0; self.pickYaw=nil else self:next() end
        end
    elseif skill=="place_in" or skill=="place_on" or skill=="place_at" then
        local dst=step.target or step.xy; local d=if type(dst)=="string" then poses[dst] else {xyz={dst[1],dst[2],0}}
        if not d then self:next(); return self.q,command,"absent_target",false end
        local height=.022
        if skill=="place_on" then
            local targetDef=nil; for _,o in ipairs(self.task.objects) do if o.id==dst then targetDef=o end end
            height=d.xyz[3]+(if targetDef then C.size(targetDef,d.size_m)[3]/2 else 0)+C.size(def,p.size_m)[3]/2+.004
        elseif skill=="place_in" then height=d.xyz[3]+.045 end
        local offset=0
        if skill=="place_in" then
            local peers={}; for _,s in ipairs(self.task.skeleton) do if s.skill=="place_in" and s.target==dst then table.insert(peers,s.obj) end end
            if #peers>1 then offset=((table.find(peers,step.obj) or 1)-(#peers+1)/2)*(C.size(def,p.size_m)[2]+.006) end
        end
        local held=M.sub(p.xyz,state.site_xyz)
        if skill=="place_at" and M.norm(held)<.06 and (not self.placeOffset or self.stage==1) then self.placeOffset=held end
        local centre=self.placeOffset
        if skill=="place_at" and centre then height=math.max(.003,C.size(def,p.size_m)[3]/2+.003-centre[3]) end
        target={d.xyz[1]-(if centre then centre[1] else 0),d.xyz[2]+offset-(if centre then centre[2] else 0),if self.stage==1 then (if skill=="place_at" then .09 else .14) else height}
        grip=-.16
        if self.stage==1 and (M.norm(M.sub(state.site_xyz,target))<.008 or self.age>6) then self:advance()
        elseif self.stage==2 and (M.norm(M.sub(state.site_xyz,target))<.006 or self.age>4) then self:next() end
    elseif skill=="push_to" then
        -- Contact can block the commanded arm. Rebase IK after a 4-degree
        -- disagreement instead of retaining an unreachable arm posture.
        if state.q then for i=1,5 do if math.abs(self.q[i]-state.q[i])>math.rad(4) then self.q=table.clone(state.q); break end end end
        local dst=step.xy; local xy=if type(dst)=="string" then poses[dst].xyz else dst
        local error={xy[1]-p.xyz[1],xy[2]-p.xyz[2],0}; local distance=M.norm(error)
        self.pushDirection=M.unit(error)
        local dir=self.pushDirection; local size=C.size(def,p.size_m)
        self.yaw=math.atan2(dir[2],dir[1])+math.pi/2
        local radius=math.sqrt(size[1]^2+size[2]^2)/2+.022
        if distance<.024 and self.stage<5 then self.stage=5; self.age=0; self.goal=table.clone(state.site_xyz) end
        if self.stage==1 then
            target={state.site_xyz[1],state.site_xyz[2],.10}
            if state.site_xyz[3]>.09 or self.age>4 then self:advance() end
        elseif self.stage==2 or self.stage==3 then
            target={p.xyz[1]-dir[1]*radius,p.xyz[2]-dir[2]*radius,if self.stage==2 then .10 else .021}
            if M.norm(M.sub(state.site_xyz,target))<.006 or self.age>5 then
                if self.stage==3 then self.pushBest=distance; self.pushProgress=0 end
                self:advance()
            end
        elseif self.stage==4 then
            local toward=M.unit(error)
            local goal=self.goal or state.site_xyz
            -- Bound command lead during contact so a blocked fingertip cannot
            -- accumulate a large sweep and launch the object on recovery.
            target={math.clamp(goal[1],state.site_xyz[1]-.012,state.site_xyz[1]+.012)+toward[1]*.00065,
                math.clamp(goal[2],state.site_xyz[2]-.012,state.site_xyz[2]+.012)+toward[2]*.00065,.021}
            local relative=M.sub(state.site_xyz,p.xyz)
            local lateral=math.abs(relative[1]*dir[2]-relative[2]*dir[1])
            if distance<(self.pushBest or distance)-.002 then self.pushBest=distance; self.pushProgress=self.age end
            if self.age-(self.pushProgress or 0)>2.25 or self.age>8 or lateral>.024 then
                self.stage=1; self.age=0; self.goal=table.clone(state.site_xyz)
            end
        else
            target={state.site_xyz[1],state.site_xyz[2],.10}
            if state.site_xyz[3]>.08 then
                if distance<.027 and (p.v or 0)<.015 and self.age>.6 then self.pushDirection=nil; self:next()
                elseif distance>=.027 and (p.v or 0)<.02 then self.stage=2; self.age=0; self.goal=table.clone(state.site_xyz) end
            end
        end
        grip=if self.stage==5 and state.site_xyz[3]>.06 then .6 else .02
    elseif skill=="release" then target=self.goal; grip=.8; if self.age>.8 then self:next() end
    elseif skill=="home" then self.q=table.clone(Robot.home_q); if self.age>1.5 then self:next() end
    elseif skill=="wait" then if self.age>=step.s then self:next() end
    end
    if target then
        self.goal=self.goal or table.clone(state.site_xyz)
        local delta=M.sub(target,self.goal); local length=M.norm(delta)
        local speed=if def and def.shape=="box" and C.size(def,p and p.size_m)[3]<.02 then .045 else .075
        if length>speed*dt then delta=M.scale(delta,speed*dt/length) end
        self.goal=M.add(self.goal,delta)
        local actualGrip=if state.q then state.q[6] else self.q[6]
        self.q[6]=actualGrip+math.clamp(grip-actualGrip,-.015,.015)
        local roll=nil
        if self.yaw then
            roll=self.yaw+self.q[1]+.06
            while roll>Robot.range[5][2] do roll-=math.pi end
            while roll<Robot.range[5][1] do roll+=math.pi end
        end
        self.q=K.ik(self.goal,self.q,roll,12)
    end
    local targetQ=S.compensate(self.q); targetQ[6]=grip
    return targetQ,command,phase,self.done
end
return S
