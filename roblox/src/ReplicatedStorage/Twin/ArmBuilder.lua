local K=require(script.Parent.Kinematics)
local M=require(script.Parent.Math3)
local Robot=require(script.Parent.Generated.Robot)
local Physics=require(script.Parent.Generated.Physics)
local JointSim=require(script.Parent.JointSim)
local A={}; A.__index=A
function A.vector(p) return Vector3.new(p[1],p[3],-p[2])*10 end
function A.unvector(v) return {v.X/10,-v.Z/10,v.Y/10} end
function A.rotation(r)
    -- C R C^T, C maps spec axes to Roblox axes. No origin shift here.
    return CFrame.new(0,0,0,r[1],r[3],-r[2],r[7],r[9],-r[8],-r[4],-r[6],r[5])
end
function A.frame(p,r,origin) return CFrame.new((origin or Vector3.zero)+A.vector(p))*A.rotation(r) end
function A.part(parent,name,size,cf,color,anchored)
    local p=Instance.new("Part"); p.Name=name; p.Size=size; p.CFrame=cf; p.Anchored=anchored or false
    p.Color=color or Color3.fromRGB(242,206,92); p.Material=Enum.Material.SmoothPlastic
    p.TopSurface=Enum.SurfaceType.Smooth; p.BottomSurface=Enum.SurfaceType.Smooth; p.Parent=parent
    return p
end
local function weld(a,b)
    local w=Instance.new("WeldConstraint"); w.Part0=a; w.Part1=b; w.Parent=b
end
function A.own(model)
    for _,p in ipairs(model:GetDescendants()) do
        if p:IsA("BasePart") and not p.Anchored then
            local ok= p:CanSetNetworkOwnership()
            if ok then p:SetNetworkOwner(nil) end
        end
    end
end
function A.new(parent,origin,q0)
    local model=Instance.new("Model"); model.Name="Arm"; model.Parent=parent
    local q=q0 or Robot.home_q; local f=K.fk(q)
    local self=setmetatable({model=model,origin=origin,roots={},hinges={},sim=JointSim.new(Physics,q),last=table.clone(q)},A)
    local templates=game:GetService("ReplicatedStorage"):WaitForChild("SO101Meshes")
    for i,b in ipairs(Robot.bodies) do
        local cf=A.frame(f[i].p,f[i].r,origin)
        -- Collision-free solver proxy. Tiny .15-stud roots ill-condition PGS and
        -- make articulated motors oscillate. Physical link/rotor dynamics live
        -- exclusively in JointSim; these proxies provide stable contact tracking.
        local root=A.part(model,b.name,Vector3.new(1,1,1),cf,nil,b.joint==0)
        root.Transparency=1; root.CanCollide=false; root.CanTouch=false; root.RootPriority=127
        root.CustomPhysicalProperties=PhysicalProperties.new(math.clamp(b.mass,.01,100),.6,0,1,1)
        root.CollisionGroup="R2PArm"; self.roots[i]=root
        for _,box in ipairs(b.boxes) do
            local p=A.part(model,"Collision",Vector3.new(box.size[1],box.size[3],box.size[2])*10,cf*CFrame.new(A.vector(box.pos)))
            p.Transparency=1; p.Massless=true; p.CollisionGroup="R2PArm"; p.CanTouch=false
            p.CustomPhysicalProperties=PhysicalProperties.new(.01,1.2,0,100,1); weld(root,p)
        end
        for _,visual in ipairs(b.visuals) do
            local template=templates:FindFirstChild(visual.asset)
            if template then
                local mesh=template:Clone(); mesh.Name="Visual"; mesh.Anchored=false; mesh.Massless=true
                mesh.CanCollide=false; mesh.CanTouch=false; mesh.CollisionGroup="R2PArm"
                mesh.CFrame=cf*CFrame.new(A.vector(visual.center)); mesh.Color=if visual.dark then Color3.fromRGB(53,60,68) else Color3.fromRGB(242,206,92)
                mesh.Material=Enum.Material.SmoothPlastic; mesh.Parent=model; weld(root,mesh)
                mesh:SetAttribute("Body",b.name)
                mesh:SetAttribute("Paintable",not visual.dark)
            end
        end
        if b.joint>0 then
            local parentRoot=self.roots[b.parent]
            local ax=b.axis; local up=if math.abs(ax[1])<.8 then {1,0,0} else {0,1,0}
            local y=M.unit(M.sub(up,M.scale(ax,M.dot(up,ax)))); local z=M.cross(ax,y)
            local basis={ax[1],y[1],z[1],ax[2],y[2],z[2],ax[3],y[3],z[3]}
            local a0=Instance.new("Attachment"); a0.Name="JointParent"
            a0.CFrame=A.frame(M.add(b.pos,M.mv(b.rot,b.joint_pos)),M.mm(b.rot,basis)); a0.Parent=parentRoot
            local a1=Instance.new("Attachment"); a1.Name="JointChild"; a1.CFrame=A.frame(b.joint_pos,basis); a1.Parent=root
            local hinge=Instance.new("HingeConstraint"); hinge.Name=Robot.joints[b.joint]
            hinge.Attachment0=a0; hinge.Attachment1=a1; hinge.ActuatorType=Enum.ActuatorType.Servo
            hinge.ServoMaxTorque=self.sim.torque[b.joint]*100; hinge.AngularSpeed=20; hinge.AngularResponsiveness=50
            hinge.TargetAngle=math.deg(q[b.joint])
            hinge.LimitsEnabled=true; hinge.LowerAngle=math.deg(Robot.range[b.joint][1]); hinge.UpperAngle=math.deg(Robot.range[b.joint][2])
            hinge.Parent=model; self.hinges[b.joint]=hinge
        end
    end
    A.own(model); return self
end
function A:measured()
    local q={}; for i,h in ipairs(self.hinges) do q[i]=math.rad(h.CurrentAngle) end; return q
end
function A:command(q) self.last=table.clone(q); self.sim:command(q) end
function A:step(dt)
    self.sim:step(dt)
    for i,h in ipairs(self.hinges) do
        h.TargetAngle=math.deg(self.sim.q[i])
    end
end
function A:tips()
    local a=self.roots[Robot.fixed_tip[1]].CFrame:PointToWorldSpace(A.vector(Robot.fixed_tip[2]))
    local b=self.roots[Robot.moving_tip[1]].CFrame:PointToWorldSpace(A.vector(Robot.moving_tip[2]))
    return A.unvector(a-self.origin),A.unvector(b-self.origin),A.unvector((a+b)/2-self.origin)
end
function A:reset(q)
    local f=K.fk(q); self.sim=JointSim.new(Physics,q); self.last=table.clone(q)
    for _,hinge in ipairs(self.hinges) do hinge.Enabled=false end
    for i,root in ipairs(self.roots) do root.CFrame=A.frame(f[i].p,f[i].r,self.origin); root.AssemblyLinearVelocity=Vector3.zero; root.AssemblyAngularVelocity=Vector3.zero end
    for i,hinge in ipairs(self.hinges) do hinge.TargetAngle=math.deg(q[i]); hinge.Enabled=true end
    A.own(self.model) -- reconnecting constraints may restore automatic ownership
end
return A
