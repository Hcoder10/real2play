local J=require("../src/ReplicatedStorage/Twin/JointSim")
local K=require("../src/ReplicatedStorage/Twin/Kinematics")
local args=require("../out/parity_inputs")
local sim=J.new(args.physics,args.q0)
local q,jaw={},{}
local function vector(v) local t={} for _,n in ipairs(v) do table.insert(t,string.format("%.15g",n)) end return "["..table.concat(t,",").."]" end
for _,target in ipairs(args.commands) do sim:command(target); sim:step(1/30); local _,_,mid=K.tips(sim.q); table.insert(q,vector(sim.q)); table.insert(jaw,vector(mid)) end
print('{"q":['..table.concat(q,",")..'],"jaw":['..table.concat(jaw,",")..'],"poses":{}}')
