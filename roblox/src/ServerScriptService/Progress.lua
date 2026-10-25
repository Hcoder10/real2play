local D=game:GetService("DataStoreService")
local Run=game:GetService("RunService")
local G=require(game:GetService("ReplicatedStorage").Twin.Generated.Game)
local Score=require(game:GetService("ReplicatedStorage").Twin.Score)
local Config=require(game:GetService("ServerStorage").TwinConfig)
local P={profiles={},top={},enabled=not Run:IsStudio() or Config.STUDIO_DATASTORES}
function P.load(player)
    if P.profiles[player.UserId] then return P.profiles[player.UserId] end
    local stats=Score.new(); stats.day=os.date("!%Y-%m-%d")
    -- Install one shared profile before the yielding read. A ready message or
    -- quick first round must not trigger a second load and replace live scores.
    P.profiles[player.UserId]=stats
    if P.enabled then
        local ok,data=pcall(function() return D:GetDataStore("R2P_Progress_v1"):GetAsync(tostring(player.UserId)) end)
        if ok and type(data)=="table" then
            stats.points+=data.points or 0
            for level,stars in pairs(data.best or {}) do stats.best[level]=math.max(stats.best[level] or 0,stars) end
            stats.stars=0; for _,stars in pairs(stats.best) do stats.stars+=stars end
            if data.day==stats.day then stats.rounds+=data.rounds or 0; stats.sent+=data.sent or 0 end
        end
    end
    return stats
end
function P.get(player) return P.profiles[player.UserId] or P.load(player) end
function P.sessionLeaders()
    local entries={}
    for _,player in ipairs(game:GetService("Players"):GetPlayers()) do
        local profile=P.profiles[player.UserId]
        if profile then table.insert(entries,{key=tostring(player.UserId),name=player.DisplayName,value=profile.points}) end
    end
    table.sort(entries,function(a,b) return if a.value==b.value then a.key<b.key else a.value>b.value end)
    while #entries>5 do table.remove(entries) end
    return entries
end
function P.save(player)
    local stats=P.profiles[player.UserId]; if not stats or not P.enabled then return end
    local snapshot=table.clone(stats)
    pcall(function() D:GetDataStore("R2P_Progress_v1"):UpdateAsync(tostring(player.UserId),function(old)
        old=old or {}; old.points=math.max(old.points or 0,snapshot.points); old.stars=math.max(old.stars or 0,snapshot.stars)
        old.best=old.best or {}; for level,stars in pairs(snapshot.best) do old.best[level]=math.max(old.best[level] or 0,stars) end
        old.day=snapshot.day; old.rounds=snapshot.rounds; old.sent=snapshot.sent; return old
    end) end)
end
function P.record(player,level,success,elapsed,limit,grabs,mode,result)
    local stats=P.get(player); local today=os.date("!%Y-%m-%d")
    if stats.day~=today then stats.day=today; stats.rounds=0; stats.sent=0 end
    local reward=Score.round(stats,level,success,elapsed,limit,grabs,result)
    if P.enabled and reward.points>0 then task.spawn(function()
        local keys={G.leaderboards.all,G.leaderboards.level..level}
        if mode=="daily" then table.insert(keys,G.leaderboards.daily..today) end
        if mode=="tic_tac_toe" then table.insert(keys,G.leaderboards.robot) end
        for _,key in ipairs(keys) do pcall(function() D:GetOrderedDataStore(key):IncrementAsync(tostring(player.UserId),reward.points) end) end
        P.save(player)
    end) end
    return reward,stats
end
function P.leaders()
    if not P.enabled then
        local entries={}
        for id,profile in pairs(P.profiles) do
            local player=game:GetService("Players"):GetPlayerByUserId(id)
            table.insert(entries,{key=tostring(id),name=player and player.DisplayName or "Teacher",value=profile.points})
        end
        table.sort(entries,function(a,b) return a.value>b.value end)
        P.top=entries; return entries
    end
    local ok,pages=pcall(function() return D:GetOrderedDataStore(G.leaderboards.all):GetSortedAsync(false,5) end)
    if ok then
        P.top=pages:GetCurrentPage()
        for _,entry in ipairs(P.top) do
            local named,name=pcall(function() return game:GetService("Players"):GetNameFromUserIdAsync(tonumber(entry.key)) end)
            if named then entry.name=name end
        end
    end
    return P.top
end
return P
