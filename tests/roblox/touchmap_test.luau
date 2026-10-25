local T=require("../../roblox/src/ReplicatedStorage/Twin/TouchMap")
local n=0
local function near(a,b) n+=1; assert(math.abs(a-b)<1e-9) end
for _,look in ipairs({{1,0},{0,1},{-1,0},{0,-1}}) do
    local up=T.map(0,1,0,0,look)
    near(up.move[1],look[1]); near(up.move[2],look[2])
    local right=T.map(1,0,0,0,look)
    near(right.move[1],look[2]); near(right.move[2],-look[1])
    near(T.map(0,0,2,-2,look).move[3],1); near(T.map(0,0,0,-2,look).roll,-1)
end
local x,y=T.stick(.05,.05); near(x,0); near(y,0)
x,y=T.stick(1,1); near(math.sqrt(x*x+y*y),1)
assert(T.sanitize({move={0/0,0,0}})==nil)
assert(T.sanitize({move={0,0,0},roll=math.huge})==nil)
assert(T.sanitize({move={5,-5,0},grip=true}).move[1]==1)
print("touch mapping: "..n.." numeric checks + input validation passed")
