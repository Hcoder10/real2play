-- Exact geometric semantics of r2p/tasks/check.py. Tests replay its shared JSON vectors.
local C={}
local function finite(x) return type(x)=="number" and x==x and math.abs(x)<math.huge end
local function vec(x,n)
    assert(type(x)=="table" and #x==n,"invalid vector")
    for _,v in ipairs(x) do assert(finite(v),"invalid number") end
    return x
end
local function norm(v) local n=0 for _,x in ipairs(v) do assert(finite(x)); n+=x*x end return math.sqrt(n) end
local function dist(a,b,n) local s=0 for i=1,n do s+=(a[i]-b[i])^2 end return math.sqrt(s) end
function C.size(o,size)
    local s=size or o.size_m
    if o.shape=="sphere" then return {s[1],s[1],s[1]} end
    if o.shape=="stick" then return {s[1],s[2],s[2]} end
    if o.shape=="bowl" then return {s[1],s[1],s[2]} end
    if o.shape=="cylinder" then
        local d={s[1],s[1],s[1]}; d[if o.axis=="x" then 1 elseif o.axis=="y" then 2 else 3]=s[2]; return d
    end
    return {s[1],s[2],s[3]}
end
local function quat(p)
    local q=p.quat
    if not p.xyz and #p==7 then q={p[4],p[5],p[6],p[7]} end
    if q then vec(q,4); local n=norm(q); assert(n>0); return {q[1]/n,q[2]/n,q[3]/n,q[4]/n} end
    return nil
end
function C.pose(p)
    assert(type(p)=="table")
    local xyz=p.xyz or (if #p==7 then {p[1],p[2],p[3]} else p)
    vec(xyz,3)
    local yaw=p.yaw_deg
    if yaw==nil then
        local q=quat(p)
        yaw=if q then math.deg(math.atan2(2*(q[1]*q[4]+q[2]*q[3]),1-2*(q[3]^2+q[4]^2))) else 0
    end
    assert(finite(yaw)); return xyz,yaw
end
local function refs(g)
    local r={}
    if g.a then table.insert(r,g.a) end; if g.b then table.insert(r,g.b) end
    for _,id in ipairs(g.objs or {}) do table.insert(r,id) end
    if type(g.xy)=="string" then table.insert(r,g.xy) end
    return r
end
function C.check(task,state)
    local objects=state.objects or {}; local defs={}
    for _,o in ipairs(task.objects) do defs[o.id]=o end
    local grip=table.clone(state.gripper or {})
    if state.gripper_open~=nil then grip.open=state.gripper_open end
    if state.site_xyz~=nil then grip.site_xyz=state.site_xyz end
    if state.holding~=nil then grip.holding=state.holding end
    local hasGrip=state.gripper_open~=nil or state.gripper~=nil
    local function geom(id)
        local p=objects[id]; local d=defs[id]
        local xyz,yaw=C.pose(p); local size=C.size(d,p.size_m)
        vec(size,3); for _,v in ipairs(size) do assert(v>0) end
        return xyz,yaw,size
    end
    local function over(a,b,margin)
        local pa=geom(a); local pb,yaw,d=geom(b)
        local dx,dy=pa[1]-pb[1],pa[2]-pb[2]; local shape=defs[b].shape
        if shape=="bowl" or shape=="cylinder" or shape=="sphere" then return math.sqrt(dx*dx+dy*dy)<d[1]/2-margin end
        local angle=math.rad(yaw); local c,s=math.cos(angle),math.sin(angle)
        return math.abs(c*dx+s*dy)<d[1]/2-margin and math.abs(-s*dx+c*dy)<d[2]/2-margin
    end
    local function onTop(a,b)
        local pa,_,sa=geom(a); local pb,_,sb=geom(b)
        return over(a,b,0) and math.abs(pa[3]-sa[3]/2-pb[3]-sb[3]/2)<.012
    end
    local function evaluate(g)
        local pred=g.pred
        if pred=="gripper_open" then return grip.open==true end
        if pred=="stacked" then
            if #g.objs<2 then return false end
            for i=2,#g.objs do if not onTop(g.objs[i],g.objs[i-1]) then return false end end
            return true
        end
        local a=g.a; local pa,yaw,sa=geom(a)
        if pred=="inside" then
            local pb,_,sb=geom(g.b)
            return over(a,g.b,.004) and pa[3]<pb[3]+sb[3]/2+sa[3]/2
        elseif pred=="on_top" then return onTop(a,g.b)
        elseif pred=="at" then local xy=if type(g.xy)=="string" then geom(g.xy) else g.xy; return dist(pa,xy,2)<g.tol_m
        elseif pred=="near" then return dist(pa,geom(g.b),2)<g.d_m
        elseif pred=="lifted" then return pa[3]-sa[3]/2>g.h_m
        elseif pred=="grasped" then
            if grip.holding~=nil then
                if type(grip.holding)=="string" then return string.find(grip.holding,a,1,true)~=nil end
                return table.find(grip.holding,a)~=nil or grip.holding[a]~=nil
            end
            if grip.open~=false or grip.site_xyz==nil then return false end
            vec(grip.site_xyz,3)
            return dist(grip.site_xyz,pa,3)<math.max(unpack(sa))/2+.02 and pa[3]-sa[3]/2>.005
        elseif pred=="upright" then
            local q=quat(objects[a]); if not q then return true end
            return math.deg(math.acos(math.clamp(1-2*(q[2]^2+q[3]^2),-1,1)))<g.tol_deg
        elseif pred=="yaw" then
            local d=defs[a]; local shape=d.shape
            if shape=="sphere" or shape=="bowl" or (shape=="cylinder" and (d.axis or "z")=="z") then return true end
            local k=if shape=="stick" or shape=="cylinder" then 2 elseif math.abs(sa[1]-sa[2])<=1e-9 then 4 else 2
            local period=360/k; return math.abs((yaw-g.deg+period/2)%period-period/2)<g.tol_deg
        elseif pred=="not_moved" then
            local initial=state.initial or {}; if initial[a]==nil then return false end
            return dist(pa,C.pose(initial[a]),2)<g.tol_m
        end
        return false
    end
    local results={}; local suppress=false
    for _,g in ipairs(task.goal) do
        local r={pred=g.pred,ok=false}; local absent={}
        for _,id in ipairs(refs(g)) do if objects[id]==nil then table.insert(absent,id) end end
        if #absent>0 then r.ok=g.optional==true; r.absent=absent
        else
            local ok,value=pcall(evaluate,g)
            if ok then r.ok=value==true else r.reason="invalid or missing predicate evidence" end
        end
        if g.pred=="grasped" or g.pred=="gripper_open" then suppress=true end
        table.insert(results,r)
    end
    if hasGrip and not suppress then table.insert(results,{pred="gripper_open",ok=grip.open==true}) end
    local settled=true
    for _,g in ipairs(task.goal) do for _,id in ipairs(refs(g)) do
        local p=objects[id]
        if type(p)=="table" and p.v~=nil then
            local ok,speed=pcall(function() return if type(p.v)=="table" then norm(p.v) else math.abs(p.v) end)
            if not ok or not finite(speed) or speed>=.5 then settled=false end
        end
    end end
    table.insert(results,{pred="settled",ok=settled})
    local ok=true; for _,r in ipairs(results) do ok=ok and r.ok end
    return {ok=ok,per_predicate=results}
end
function C.check_window(task,states,hz,hold)
    hz=hz or 30; hold=hold or .5; assert(finite(hz) and hz>0 and finite(hold) and hold>0)
    local count=math.ceil(hz*hold)
    if #states==0 then return {ok=false,per_predicate={{pred="window",ok=false}},ticks=0,required_ticks=count} end
    local initial=states[1].initial or states[1].objects or {}; local checks={}
    for i=math.max(1,#states-count+1),#states do
        local state=table.clone(states[i]); state.initial=state.initial or initial
        table.insert(checks,C.check(task,state))
    end
    local results={}
    for i,g in ipairs(task.goal) do
        local ok=true; for _,c in ipairs(checks) do ok=ok and c.per_predicate[i].ok end
        table.insert(results,{pred=g.pred,ok=ok})
    end
    for _,name in ipairs({"gripper_open","settled"}) do
        local found,ok=false,true
        for _,c in ipairs(checks) do for i=#task.goal+1,#c.per_predicate do
            local p=c.per_predicate[i]; if p.pred==name then found=true; ok=ok and p.ok end
        end end
        if found then table.insert(results,{pred=name,ok=ok}) end
    end
    if #states<count then table.insert(results,{pred="window",ok=false}) end
    local ok=#states>=count; for _,r in ipairs(results) do ok=ok and r.ok end
    return {ok=ok,per_predicate=results,ticks=math.min(#states,count),required_ticks=count}
end
return C
