local A=require(script.Parent.ArmBuilder)
local C=require(script.Parent.TaskCheck)
local Scene={}
local colors={brown=Color3.fromRGB(172,116,76),blue=Color3.fromRGB(67,134,211),green=Color3.fromRGB(78,203,169),gray=Color3.fromRGB(153,174,192),red=Color3.fromRGB(231,96,99),yellow=Color3.fromRGB(242,206,92)}
local function color(value)
    if type(value)=="table" then return Color3.new(value[1],value[2],value[3]) end
    if type(value)=="string" and value:sub(1,1)=="#" then return Color3.fromHex(value:sub(2)) end
    return colors[value] or colors.gray
end
function Scene.build(parent,origin,scene)
    local model=Instance.new("Model"); model.Name="Objects"; model.Parent=parent
    local roots={}
    for _,o in ipairs(scene.objects) do
        local size=C.size(o); local cf=CFrame.new(origin+A.vector(o.xyz))*CFrame.Angles(0,math.rad(o.yaw_deg or 0),0)
        local p=A.part(model,o.id,Vector3.new(size[1],size[3],size[2])*10,cf,color(o.color),o.fixed)
        roots[o.id]=p; p.CollisionGroup="R2PObject"; p.CanCollide=o.collide~=false
        local volume=p.Size.X*p.Size.Y*p.Size.Z
        p.CustomPhysicalProperties=PhysicalProperties.new(math.clamp((o.mass_kg or .03)/volume,.01,100),o.friction or .6,0,10,1)
        if o.shape=="sphere" then p.Shape=Enum.PartType.Ball
        elseif o.shape=="cylinder" then
            p.Shape=Enum.PartType.Cylinder
            local length=o.size_m[2]*10; local diameter=o.size_m[1]*10
            p.Size=Vector3.new(length,diameter,diameter)
            p.CFrame=cf*(if o.axis=="x" then CFrame.identity elseif o.axis=="y" then CFrame.Angles(0,math.pi/2,0) else CFrame.Angles(0,0,math.pi/2))
        elseif o.shape=="bowl" then
            local wall=o.wall_m or .005; local radius=o.size_m[1]/2; local height=o.size_m[2]
            p.Size=Vector3.new((radius+wall)*20,wall*10,(radius+wall)*20)
            p.CFrame=cf*CFrame.new(0,(-height/2+wall/2)*10,0)
            -- Root origin stays at the nominal bowl centre for traces via offset attribute.
            p:SetAttribute("CenterOffset",Vector3.new(0,(height/2-wall/2)*10,0))
            for i=0,15 do
                local phi=i*math.pi/8; local r=(radius+wall/2)*10
                local w=A.part(model,o.id.."Wall",Vector3.new(wall*10,(height-wall)*10,(radius+wall)*math.tan(math.pi/16)*20),
                    cf*CFrame.new(r*math.cos(phi),wall*5,-r*math.sin(phi))*CFrame.Angles(0,phi,0),color(o.color),false)
                w.Massless=true; w.CollisionGroup="R2PObject"
                local joint=Instance.new("WeldConstraint"); joint.Part0=p; joint.Part1=w; joint.Parent=w
            end
        end
        if o.role=="target" then p.Material=Enum.Material.Neon; p.Transparency=.45 end
        if o.role=="target" or o.role=="container" then
            local glow=Instance.new("Highlight"); glow.FillTransparency=.85; glow.OutlineColor=Color3.fromRGB(78,203,169); glow.Adornee=p; glow.Parent=p
        end
    end
    A.own(model); return model,roots
end
function Scene.poses(roots,origin,definitions)
    local poses={}
    for _,o in ipairs(definitions) do
        local p=roots[o.id]
        if p and p.Parent then
            local cf=p.CFrame; local pos=cf:PointToWorldSpace(p:GetAttribute("CenterOffset") or Vector3.zero)
            local right=cf.RightVector; local yaw=math.deg(math.atan2(-right.Z,right.X))
            local axis,angle=cf:ToAxisAngle(); local s=math.sin(angle/2)
            poses[o.id]={xyz=A.unvector(pos-origin),yaw_deg=yaw,quat={math.cos(angle/2),axis.X*s,-axis.Z*s,axis.Y*s},size_m=o.size_m,v=p.AssemblyLinearVelocity.Magnitude/10}
        end
    end
    return poses
end
return Scene
