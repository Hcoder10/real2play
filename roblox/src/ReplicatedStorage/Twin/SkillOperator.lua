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
    local g=J.inverse(q,table.create(6,0),table.create(6,0),9.81); local out=table.clone(q)
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
    self.index+=1; self.stage=1; self.age=0
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
        local z=math.max(.006,xyz[3]+size[3]*.15)
        target={xyz[1],xyz[2],if self.stage==1 or self.stage==4 then .13 else z}
        grip=if self.stage<3 then .95 else -.16
        if self.stage==1 and (M.norm(M.sub(state.site_xyz,target))<.008 or self.age>2.5) then self:advance()
        elseif self.stage==2 and (M.norm(M.sub(state.site_xyz,target))<.006 or self.age>2) then self:advance()
        elseif self.stage==3 and self.age>.65 then self:advance()
        elseif self.stage==4 and self.age>1.2 then self:next() end
    elseif skill=="place_in" or skill=="place_on" or skill=="place_at" then
        local dst=step.target or step.xy; local d=if type(dst)=="string" then poses[dst] else {xyz={dst[1],dst[2],0}}
        if not d then self:next(); return self.q,command,"absent_target",false end
        local height=.022
        if skill=="place_on" then
            local targetDef=nil; for _,o in ipairs(self.task.objects) do if o.id==dst then targetDef=o end end
            height=d.xyz[3]+(if targetDef then C.size(targetDef,d.size_m)[3]/2 else 0)+C.size(def,p.size_m)[3]/2+.004
        elseif skill=="place_in" then height=d.xyz[3]+.045 end
        target={d.xyz[1],d.xyz[2],if self.stage==1 then .14 else height}
        grip=-.16
        if self.stage==1 and (M.norm(M.sub(state.site_xyz,target))<.008 or self.age>2.5) then self:advance()
        elseif self.stage==2 and (M.norm(M.sub(state.site_xyz,target))<.007 or self.age>2.5) then self:next() end
    elseif skill=="push_to" then
        local dst=step.xy; local xy=if type(dst)=="string" then poses[dst].xyz else dst
        self.pushDirection=self.pushDirection or M.unit({xy[1]-p.xyz[1],xy[2]-p.xyz[2],0})
        local dir=self.pushDirection; local size=C.size(def,p.size_m)
        target=if self.stage<3 then {p.xyz[1]-dir[1]*(size[1]/2+.025),p.xyz[2]-dir[2]*(size[2]/2+.025),if self.stage==1 then .12 else .015} else {xy[1]-.01*dir[1],xy[2]-.01*dir[2],.015}
        grip=-.16
        if self.age>(if self.stage==3 then 3 else 1.8) then if self.stage==3 then self.pushDirection=nil; self:next() else self:advance() end end
    elseif skill=="release" then grip=.95; if self.age>.6 then self:next() end
    elseif skill=="home" then self.q=table.clone(Robot.home_q); if self.age>1.5 then self:next() end
    elseif skill=="wait" then if self.age>=step.s then self:next() end
    end
    if target then self.q=K.ik(target,self.q,nil,10) end
    self.q[6]=grip
    return S.compensate(self.q),command,phase,self.done
end
return S
