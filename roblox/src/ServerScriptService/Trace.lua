local H=game:GetService("HttpService")
local Config=require(game:GetService("ServerStorage").TwinConfig)
local Trace={}; Trace.__index=Trace
function Trace.new(meta,enabled)
    meta.schema="r2p_trace/1"; meta.commands={}
    return setmetatable({meta=meta,ticks={},enabled=enabled~=false,id=(meta.source or "human").."_"..H:GenerateGUID(false),segments={},active=nil},Trace)
end
function Trace:phase(command,phase)
    if self.active and self.active.command==command then self.phaseName=phase; return end
    local t=#self.ticks/30
    if self.active then table.insert(self.segments,{self.active.start,t,self.active.command}) end
    self.active={start=t,command=command}; self.phaseName=phase
end
function Trace:tick(q,target,poses,closed,slots)
    if not self.enabled then return end
    assert(self.active,"phase event required before trace tick")
    table.insert(self.ticks,{t=#self.ticks/30,q=table.clone(q),q_target=table.clone(target),poses=poses,grip_closed=closed,
        command=self.active.command,phase=self.phaseName,slots=slots})
end
function Trace:finish(success,reason,extra)
    if self.finished then return end; self.finished=true
    if self.active then table.insert(self.segments,{self.active.start,#self.ticks/30,self.active.command}) end
    self.meta.success=success; self.meta.fail_reason=reason or ""; self.meta.commands=self.segments
    for k,v in pairs(extra or {}) do self.meta[k]=v end
end
function Trace:upload(callback)
    if not self.enabled or #self.ticks==0 then if callback then callback(false,"not_recorded") end return end
    assert(self.finished,"finish trace before uploading")
    if Config.TRACE_URL=="" or Config.TRACE_KEY=="" then
        if callback then callback(false,"trace endpoint not configured") end
        return
    end
    task.spawn(function()
        local parts=math.ceil(#self.ticks/600)
        for part=0,parts-1 do
            local ticks={}; for i=part*600+1,math.min(#self.ticks,(part+1)*600) do table.insert(ticks,self.ticks[i]) end
            local payload={trace_id=self.id,part=part,final=part==parts-1,ticks=ticks}
            if part==0 then payload.meta=self.meta end
            if part==parts-1 then payload.meta_final={success=self.meta.success,fail_reason=self.meta.fail_reason,commands=self.meta.commands} end
            local body=H:JSONEncode(payload); local delivered=false
            for attempt=1,Config.UPLOAD_RETRIES do
                local ok,response=pcall(function() return H:RequestAsync({Url=Config.TRACE_URL:gsub("/$","").."/trace",Method="POST",
                    Headers={["Content-Type"]="application/json",["X-Trace-Key"]=Config.TRACE_KEY},Body=body}) end)
                if ok and response.Success then
                    local parsed=H:JSONDecode(response.Body)
                    delivered=parsed.ok==true and parsed.part==part and (part<parts-1 or parsed.complete==true)
                    if delivered then break end
                end
                if ok and response.StatusCode>=400 and response.StatusCode<500 and response.StatusCode~=429 then break end
                task.wait(math.min(8,2^(attempt-1))+.1*math.random())
            end
            if not delivered then if callback then callback(false,"upload_failed") end return end
            -- Stay below HttpService's shared 500 requests/minute budget.
            if part<parts-1 then task.wait(.2) end
        end
        if callback then callback(true,self.id) end
    end)
end
return Trace
