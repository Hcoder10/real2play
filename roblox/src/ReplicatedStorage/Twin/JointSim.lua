--!native
-- Fixed-base recursive Newton-Euler dynamics, including the six rotor inertias.
-- No Roblox APIs: the same code is exercised against MuJoCo by the Luau runner.
local M = require(if script then script.Parent.Math3 else "./Math3")
local K = require(if script then script.Parent.Kinematics else "./Kinematics")
local Robot = require(if script then script.Parent.Generated.Robot else "./Generated/Robot")
local JointSim = {}; JointSim.__index=JointSim
local ZERO={0,0,0}
function JointSim.inverse(q,qd,qdd,gravity,frames)
    local f=frames or K.fk(q)
    local w={[0]=ZERO}; local alpha={[0]=ZERO}; local accel={[0]={0,0,gravity or 0}}
    local force,torque={},{}
    for i,b in ipairs(Robot.bodies) do
        local p=b.parent; local arm=M.sub(f[i].p,f[p].p)
        local jw=if b.joint>0 then M.scale(f[i].axis,qd[b.joint]) else ZERO
        local ja=if b.joint>0 then M.scale(f[i].axis,qdd[b.joint]) else ZERO
        w[i]=M.add(w[p],jw)
        alpha[i]=M.add(M.add(alpha[p],ja),M.cross(w[p],jw))
        accel[i]=M.add(accel[p],M.add(M.cross(alpha[p],arm),M.cross(w[p],M.cross(w[p],arm))))
        local c=M.sub(f[i].com,f[i].p)
        force[i]=M.scale(M.add(accel[i],M.add(M.cross(alpha[i],c),M.cross(w[i],M.cross(w[i],c)))),b.mass)
        local inertia=M.mm(M.mm(f[i].r,b.inertia),M.transpose(f[i].r))
        torque[i]=M.add(M.add(M.mv(inertia,alpha[i]),M.cross(w[i],M.mv(inertia,w[i]))),M.cross(c,force[i]))
    end
    local tau=table.create(6,0)
    for i=#Robot.bodies,1,-1 do
        local b=Robot.bodies[i]
        if b.joint>0 then tau[b.joint]=M.dot(f[i].axis,torque[i]) end
        if b.parent>0 then
            local p=b.parent
            force[p]=M.add(force[p],force[i])
            torque[p]=M.add(torque[p],M.add(torque[i],M.cross(M.sub(f[i].p,f[p].p),force[i])))
        end
    end
    return tau
end
function JointSim.massBias(q,qd,armature,gravity)
    local f=K.fk(q); local z=table.create(6,0)
    local bias=JointSim.inverse(q,qd,z,gravity or 9.81,f)
    local mass=table.create(6)
    for i=1,6 do mass[i]=table.create(6,0) end
    for j=1,6 do
        local unit=table.create(6,0); unit[j]=1
        local col=JointSim.inverse(q,z,unit,0,f)
        for i=1,6 do mass[i][j]=col[i] end
        mass[j][j]+=armature[j]
    end
    return mass,bias
end
local function array(value,default)
    if type(value)=="table" then return table.clone(value) end
    return table.create(6,value or default)
end
function JointSim.new(physics,q)
    local c=physics.config or physics; local servo=c.servo or c
    local self=setmetatable({q=table.clone(q or Robot.home_q),v=table.create(6,0),accumulator=0,queue={}},JointSim)
    self.kp=array(servo.kp,17.8); self.kv=array(servo.kv,0)
    self.damping=array(servo.damping,0.6); self.armature=array(servo.armature,0.028)
    self.friction=array(servo.frictionloss,0.052); self.torque=array(servo.torque_limit or servo.torque_limits,3.35)
    self.sag=array(c.gravity_sag_gain or c.sag_gain or c.gravity_sag_k or 0,0)
    if c.joints then
        for i,name in ipairs(Robot.joints) do
            local j=assert(c.joints[name],"missing physics joint "..name)
            self.kp[i]=j.kp; self.kv[i]=j.kv; self.damping[i]=j.damping or 0
            self.armature[i]=j.armature; self.friction[i]=j.frictionloss; self.torque[i]=j.torque_limit
            -- Reference twin's sag is the equilibrium of finite servo stiffness
            -- and gravity. sag_gain describes the real compensation model only.
            self.sag[i]=0
        end
    end
    self.latency=c.latency_ticks or 0
    self.gravity=if c.gravity then -c.gravity[3] else 9.81
    self.target=table.clone(self.q)
    return self
end
function JointSim:command(q)
    local target={}
    for i=1,6 do
        assert(type(q[i])=="number" and q[i]==q[i] and math.abs(q[i])<1e6,"invalid joint target")
        target[i]=math.clamp(q[i],Robot.range[i][1],Robot.range[i][2])
    end
    table.insert(self.queue,target)
    if #self.queue>self.latency then self.target=table.remove(self.queue,1) end
end
function JointSim:substep(h)
    local mass,bias=JointSim.massBias(self.q,self.v,self.armature,self.gravity)
    local gravity=JointSim.inverse(self.q,table.create(6,0),table.create(6,0),self.gravity)
    local rhs={}
    for i=1,6 do
        local wanted=self.target[i]-self.sag[i]*gravity[i]
        local tau=math.clamp(self.kp[i]*(wanted-self.q[i])-self.kv[i]*self.v[i],-self.torque[i],self.torque[i])
        rhs[i]=tau-bias[i]-self.damping[i]*self.v[i]
        -- Regularized Coulomb friction, bounded so it cannot reverse a resting joint.
        local stop=self.v[i]*mass[i][i]/h+rhs[i]
        rhs[i]-=math.clamp(stop,-self.friction[i],self.friction[i])
        mass[i][i]+=h*(self.damping[i]+self.kv[i])
    end
    local acceleration=M.solve(mass,rhs)
    for i=1,6 do
        self.v[i]+=h*acceleration[i]
        self.q[i]+=h*self.v[i]
        local low,high=Robot.range[i][1],Robot.range[i][2]
        if self.q[i]<low then self.q[i]=low; self.v[i]=math.max(0,self.v[i]) end
        if self.q[i]>high then self.q[i]=high; self.v[i]=math.min(0,self.v[i]) end
    end
end
function JointSim:step(dt)
    self.accumulator+=dt
    while self.accumulator>=1/480-1e-10 do self:substep(1/480); self.accumulator-=1/480 end
    return self.q
end
return JointSim
