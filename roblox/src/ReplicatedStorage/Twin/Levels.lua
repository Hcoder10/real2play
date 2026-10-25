local Data=require(if script then script.Parent.Generated.Tasks else "./Generated/Tasks")
local Check=require(if script then script.Parent.TaskCheck else "./TaskCheck")
local L={order=Data.order,tasks=Data.tasks}
function L.copy(t) if type(t)~="table" then return t end local out={} for k,v in pairs(t) do out[k]=L.copy(v) end return out end
function L.random(seed)
    local state=math.max(1,math.floor(seed)%2147483647)
    return function() state=state*16807%2147483647; return (state-1)/2147483646 end
end
function L.commands(task)
    local out={}
    local fields={pick={"obj"},place_in={"obj","target"},place_on={"obj","target"},place_at={"obj","xy"},push_to={"obj","xy"},release={},home={},wait={"s"}}
    for _,step in ipairs(task.skeleton) do
        local args={}
        for _,key in ipairs(fields[step.skill]) do local v=step[key]; table.insert(args,if type(v)=="table" then "["..table.concat(v,",").."]" else tostring(v)) end
        table.insert(out,step.skill..(if #args>0 then "("..table.concat(args,",")..")" else ""))
    end
    return out
end
local function radius(o)
    local s=Check.size(o)
    return if o.shape=="sphere" or o.shape=="bowl" or (o.shape=="cylinder" and (o.axis or "z")=="z") then s[1]/2+(o.wall_m or 0) else math.sqrt(s[1]^2+s[2]^2)/2
end
function L.sample(task,seed)
    local rng=L.random(seed); local rnd=task.randomize or {}
    local function draw(x) return if type(x)=="table" then x[1]+rng()*(x[2]-x[1]) else x end
    local active={}
    -- Presence is sampled once, independently of geometric rejection.
    for _,o in ipairs(task.objects) do if rng()<((rnd[o.id] or {}).p_present or 1) then table.insert(active,o) end end
    for _=1,500 do
        local placed={}; local pending=table.clone(active); local failed=false
        while #pending>0 do
            local index=nil; for i,o in ipairs(pending) do if not o.on or placed[o.on] then index=i; break end end
            assert(index,"absent or cyclic object support")
            local o=L.copy(table.remove(pending,index)); local rule=rnd[o.id] or {}
            local size={}; for i,d in ipairs(rule.size_m or o.size_m) do size[i]=draw(d)*draw(rule.size_scale or 1) end; o.size_m=size
            o.mass_kg=draw(rule.mass_kg or o.mass_kg or .03); o.friction=draw(rule.friction or o.friction or .6)
            local x,y
            if rule.xy_choice then local p=rule.xy_choice[math.min(#rule.xy_choice,math.floor(rng()*#rule.xy_choice)+1)]; x,y=p[1],p[2]
            elseif rule.r then local r,az=draw(rule.r),math.rad(draw(rule.az_deg)); x,y=r*math.cos(az),r*math.sin(az)
            elseif rule.x then x,y=draw(rule.x),draw(rule.y)
            elseif o.on then x,y=placed[o.on].xyz[1],placed[o.on].xyz[2]
            else local p=o.xy or o.xyz; x,y=p[1],p[2] end
            local yaw=draw(rule.yaw_deg or o.yaw_deg or 0)
            if rule.yaw_rel_deg then yaw=math.deg(math.atan2(y,x))+draw(rule.yaw_rel_deg) end
            local h=Check.size(o)[3]; local z=if o.xyz then o.xyz[3] else h/2
            if o.on then local base=placed[o.on]; z=base.xyz[3]+Check.size(base)[3]/2+h/2 end
            o.xyz={x,y,z}; o.yaw_deg=(yaw+180)%360-180; o.xy=nil
            if o.fixed==nil then o.fixed=o.role=="fixture" end
            if o.shape=="bowl" then o.wall_m=o.wall_m or .005; o.container=true end
            for _,other in pairs(placed) do
                local distance=math.sqrt((x-other.xyz[1])^2+(y-other.xyz[2])^2)
                local discrete=rule.xy_choice and (rnd[other.id] or {}).xy_choice
                local overlap=if discrete then distance<1e-9 else o.collide~=false and other.collide~=false and o.on~=other.id and distance<radius(o)+radius(other)+(rnd.min_sep_m or .03)
                if overlap then failed=true; break end
            end
            local tab=task.table or {}; local center=tab.center_xy or {.2,0}; local dims=tab.size_m or {.8,.6}
            if math.abs(x-center[1])+radius(o)>dims[1]/2 or math.abs(y-center[2])+radius(o)>dims[2]/2 then failed=true end
            if failed then break end
            placed[o.id]=o
        end
        if not failed then
            local objects={}; for _,o in ipairs(active) do table.insert(objects,placed[o.id]) end
            return {version=0,task=task.name,table=L.copy(task.table or {size_m={.8,.6},center_xy={.2,0}}),objects=objects}
        end
    end
    error("cannot sample non-overlapping scene: "..task.name)
end
function L.slots(task,poses)
    local slots={}
    for i=1,task.limits.max_objects do
        local o=task.objects[i]; local p=o and poses[o.id]
        if p then
            local xyz,yaw=Check.pose(p); if o.shape=="stick" then yaw=(yaw+90)%180-90 end
            local s=Check.size(o,p.size_m); local a=math.rad(yaw)
            slots[i]={xyz[1],xyz[2],xyz[3],math.sin(a),math.cos(a),s[1],s[2],s[3],1}
        else slots[i]=table.create(9,0) end
    end
    return slots
end
return L
