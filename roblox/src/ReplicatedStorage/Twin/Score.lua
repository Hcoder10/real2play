local G=require(if script then script.Parent.Generated.Game else "./Generated/Game")
local S={}
function S.new() return {points=0,stars=0,combo=0,streak=0,rounds=0,sent=0,best={}} end
function S.round(stats,level,success,elapsed,limit,grabs,result)
    stats.rounds+=1
    if not success then stats.combo=0; stats.streak=0; return {points=0,stars=0,combo=0} end
    stats.combo=math.min(G.scoring.combo_cap,stats.combo+1); stats.streak+=1
    local fraction=math.clamp(elapsed/limit,0,1)
    local stars=1+(if fraction<=G.stars.time_fraction then 1 else 0)+(if grabs<=G.stars.clean_grabs then 1 else 0)
    local points=math.floor((G.scoring.base+G.scoring.speed*(1-fraction)+(if grabs<=1 then G.scoring.clean else 0))*stats.combo)
    if result~=nil then points=if result==1 then G.scoring.win elseif result==0 then G.scoring.draw else 0 end
    stats.points+=points
    local old=stats.best[level] or 0; stats.best[level]=math.max(old,stars); stats.stars+=stats.best[level]-old
    return {points=points,stars=stars,combo=stats.combo}
end
function S.unlocks(stars) local out={} for _,u in ipairs(G.unlocks) do if stars>=u.stars then table.insert(out,u) end end return out end
function S.daily(day) local hash=0 for i=1,#day do hash=(hash*31+string.byte(day,i))%2147483647 end return hash end
return S
