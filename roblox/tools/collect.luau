local args=_G.R2P_ARGS or {}
local Config=require(game:GetService("ServerStorage").TwinConfig)
Config.BOT_IDLE=false
if args.trace_url then Config.TRACE_URL=args.trace_url; Config.TRACE_KEY=args.trace_key end
local deadline=os.clock()+60
while not _G.R2P_HARNESS and os.clock()<deadline do task.wait(.1) end
local H=assert(_G.R2P_HARNESS)
local L=require(game:GetService("ReplicatedStorage").Twin.Levels)
local names=args.tasks or L.order
local results={}
for _,name in ipairs(names) do for round=1,args.rounds or 1 do
    local station=H.stations[1]; station:start(name,(args.seed or 100)+round,"bot",{limit=args.limit or 50})
    local phases={}; local last=nil; deadline=os.clock()+90
    while station.active and os.clock()<deadline do
        local phase=station.trace.phaseName
        if phase~=last then table.insert(phases,{phase=phase,t=station.elapsed,site=station:state().site_xyz,objects=station:state().objects}); last=phase end
        task.wait(.1)
    end
    if station.active then station:finish(false,"watchdog") end
    local trace=station.trace
    table.insert(results,{task=name,seed=station.seed,success=trace.meta.success,reason=trace.meta.fail_reason,ticks=#trace.ticks,trace_id=trace.id,phases=phases,final=station:state().objects})
    station:sleep(true)
end end
task.wait(1)
return {results=results}
