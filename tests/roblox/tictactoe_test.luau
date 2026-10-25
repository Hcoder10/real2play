local T=require("../../roblox/src/ReplicatedStorage/Twin/TicTacToe")
local n=0
local function check(x,msg) n+=1; assert(x,msg) end
local board=T.new(); check(T.place(board,1,1)); check(not T.place(board,1,2)); check(not T.place(board,3,1)); check(T.place(board,5,2))
local function walk(b,turn)
    local result=T.result(b)
    if result~=nil then check(result~=1,"perfect robot lost"); return end
    if turn==2 then
        local c=T.choose(b,0,function() return .5 end); check(c~=nil and b[c]==0)
        local copy=table.clone(b); check(T.place(copy,c,2)); walk(copy,1)
    else
        for _,c in ipairs(T.legal(b)) do local copy=table.clone(b); check(T.place(copy,c,1)); walk(copy,2) end
    end
end
walk(T.new(),1)
local seen={}
local function exhaustive(b,turn)
    local key=table.concat(b)..turn; if seen[key] then return end; seen[key]=true
    if T.result(b)~=nil then check(#T.legal(b)==0); return end
    if turn==2 then
        local wins={}
        for _,c in ipairs(T.legal(b)) do b[c]=2; if T.result(b)==2 then wins[c]=true end; b[c]=0 end
        local chosen=T.choose(b,1,function() return 0 end)
        check(chosen~=nil and b[chosen]==0)
        if next(wins) then check(wins[chosen],"slip skipped an immediate win") end
    end
    for _,c in ipairs(T.legal(b)) do local copy=table.clone(b); check(T.place(copy,c,turn)); exhaustive(copy,3-turn) end
end
exhaustive(T.new(),1)
for i=1,9 do local p=T.xy(i); check(T.cell({p[1],p[2],.007},T.new())==i) end
check(T.cell({.1,.1,0},T.new())==nil)
print("tic-tac-toe: "..n.." checks; exhaustive reachable game tree and perfect-play defense passed")
