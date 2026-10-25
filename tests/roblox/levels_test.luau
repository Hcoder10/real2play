local L=require("../../roblox/src/ReplicatedStorage/Twin/Levels")
local S=require("../../roblox/src/ReplicatedStorage/Twin/Score")
local C=require("../../roblox/src/ReplicatedStorage/Twin/TaskCheck")
local Skill=require("../../roblox/src/ReplicatedStorage/Twin/SkillOperator")
local K=require("../../roblox/src/ReplicatedStorage/Twin/Kinematics")
local Robot=require("../../roblox/src/ReplicatedStorage/Twin/Generated/Robot")
local n=0
assert(#L.order>=6)
for _,name in ipairs(L.order) do
    local task=L.tasks[name]; assert(#L.commands(task)==#task.skeleton)
    for seed=1,30 do
        local scene=L.sample(task,seed); local again=L.sample(task,seed); local poses={}
        assert(#scene.objects==#again.objects)
        for i,o in ipairs(scene.objects) do
            assert(o.xyz[1]==again.objects[i].xyz[1]); poses[o.id]={xyz=o.xyz,yaw_deg=o.yaw_deg,size_m=o.size_m}; n+=1
            if name=="pick_place_stick" and o.id=="stick" then
                local original=task.objects[1].size_m
                assert(math.abs(o.size_m[1]/original[1]-o.size_m[2]/original[2])<1e-10,"size_scale must preserve aspect ratio")
                n+=1
            end
        end
        local slots=L.slots(task,poses); assert(#slots==task.limits.max_objects)
        for _,slot in ipairs(slots) do assert(#slot==9); n+=1 end
        local op=Skill.new(task); local _,_,site=K.tips(Robot.home_q)
        local q,command=op:step({objects=poses,site_xyz=site},1/30)
        assert(#q==6 and type(command)=="string")
    end
end
local release=Skill.new(L.tasks.place_on_cell)
release.index=3; release.releaseHold=true; release.q={-.8,-.2,.7,1.0,.8,.2}
local fixed,_,mid=K.tips(release.q); release.goal=mid
local target,command=release:step({objects={},site_xyz=mid,q=release.q},1/30)
local after=K.tips(release.q)
assert(command=="release" and target[6]>.15,"release must open the jaw")
for axis=1,3 do assert(math.abs(after[axis]-fixed[axis])<1e-10,"opening must not translate the fixed finger into the placed piece"); n+=1 end
local pinch=Skill.new(L.tasks.place_on_cell)
pinch.stage=3; pinch.q={-.4,-.2,.7,1.0,.8,.55}
local beforePinch,_,pinchSite=K.tips(pinch.q)
pinch:step({objects={piece={xyz={.218,-.075,.0085},size_m={.033,.017},yaw_deg=0}},site_xyz=pinchSite,q=table.clone(pinch.q)},1/30)
local afterPinch=K.tips(pinch.q)
for axis=1,3 do assert(math.abs(afterPinch[axis]-beforePinch[axis])<1e-10,"closing must not chase the changing jaw midpoint and shove the cap"); n+=1 end
local stats=S.new()
local result=S.round(stats,"a",true,10,40,1)
assert(result.stars==3 and stats.stars==3 and stats.combo==1)
S.round(stats,"a",true,30,40,3); assert(stats.stars==3 and stats.combo==2)
S.round(stats,"b",false,40,40,3); assert(stats.combo==0 and stats.streak==0)
assert(S.round(stats,"ttt",true,30,40,1,1).points==300)
assert(S.round(stats,"ttt",true,30,40,1,0).points==100)
assert(#S.unlocks(16)==5 and #S.unlocks(1)==0)
assert(S.daily("2026-10-25")==S.daily("2026-10-25") and S.daily("2026-10-25")~=S.daily("2026-10-26"))
print("levels/scoring: "..n.." scene/slot checks, all task operators, stars, combos, daily seed and unlocks passed")
