local J=require("../../roblox/src/ReplicatedStorage/Twin/JointSim")
local K=require("../../roblox/src/ReplicatedStorage/Twin/Kinematics")
local cases=require("./dynamics_vectors")
local n=0
local function near(a,b,tol) assert(math.abs(a-b)<tol,string.format("%.12g != %.12g",a,b)); n+=1 end
for _,c in ipairs(cases) do
    local a,b=K.tips(c.q)
    for i=1,3 do near(a[i],c.fixed[i],1e-9); near(b[i],c.moving[i],1e-9) end
    local mass,bias=J.massBias(c.q,c.v,c.armature)
    for i=1,6 do
        near(bias[i],c.bias[i],1e-9)
        for j=1,6 do near(mass[i][j],c.mass[i][j],1e-9) end
    end
end
print("dynamics: "..n.." FK / mass / bias checks passed")
