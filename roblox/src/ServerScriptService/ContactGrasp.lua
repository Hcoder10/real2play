-- Contact-confirmed rigid grasp for light caps. Never acquire by proximity.
-- GetPartsInPart tests actual collider volumes on both fingers; the weld keeps
-- the existing object-to-gripper transform and never assigns an object pose.
local Robot=require(game:GetService("ReplicatedStorage").Twin.Generated.Robot)
local G={}; G.__index=G
function G.new(arm)
    local parts={}; for part in pairs(arm.fingerSides) do table.insert(parts,part) end
    local params=OverlapParams.new(); params.FilterType=Enum.RaycastFilterType.Include; params.FilterDescendantsInstances=parts
    return setmetatable({arm=arm,params=params,held=nil,candidate=nil,contacts=0},G)
end
function G:release()
    if self.held then
        local part=self.held.part; self.held.weld:Destroy(); self.held=nil
        if part.Parent and part:CanSetNetworkOwnership() then part:SetNetworkOwner(nil) end
    end
    self.candidate=nil; self.contacts=0
end
function G:update(closed,roots,definitions)
    if not closed then self:release(); return nil end
    if self.held then
        if self.held.part.Parent then return self.held.q end
        self:release()
    end
    local found=nil
    for _,definition in ipairs(definitions) do
        local part=roots[definition.id]
        if definition.role=="manipulated" and part and not part.Anchored then
            local fixed,moving=false,false
            for _,contact in ipairs(workspace:GetPartsInPart(part,self.params)) do
                local side=self.arm.fingerSides[contact]
                if side=="fixed" then fixed=true elseif side=="moving" then moving=true end
            end
            if fixed and moving then found=part; break end
        end
    end
    if found and found==self.candidate then self.contacts+=1 else self.candidate=found; self.contacts=if found then 1 else 0 end
    if self.contacts<2 then return nil end
    local q=self.arm:measured()[6]
    local weld=Instance.new("WeldConstraint"); weld.Name="ContactConfirmedCapGrip"
    weld.Part0=self.arm.roots[Robot.fixed_tip[1]]; weld.Part1=found; weld.Parent=self.arm.model
    if found:CanSetNetworkOwnership() then found:SetNetworkOwner(nil) end
    self.held={part=found,weld=weld,q=q}
    return q
end
return G
