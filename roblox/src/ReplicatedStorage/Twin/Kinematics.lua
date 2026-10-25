--!native
local M = require(if script then script.Parent.Math3 else "./Math3")
local Robot = require(if script then script.Parent.Generated.Robot else "./Generated/Robot")
local K = {}
function K.fk(q, robot)
    robot=robot or Robot
    local f={[0]={p=M.scale(robot.base_origin,-1),r=M.eye()}}
    for i,b in ipairs(robot.bodies) do
        local parent=f[b.parent]
        local r=M.mm(parent.r,b.rot)
        local p=M.add(parent.p,M.mv(parent.r,b.pos))
        local joint=M.add(p,M.mv(r,b.joint_pos))
        local axis=M.mv(r,b.axis)
        if b.joint>0 then
            r=M.mm(r,M.rotation(b.axis,q[b.joint]))
            p=M.sub(joint,M.mv(r,b.joint_pos))
        end
        f[i]={p=p,r=r,joint=joint,axis=axis,com=M.add(p,M.mv(r,b.com))}
    end
    return f
end
function K.point(f,tip) return M.add(f[tip[1]].p,M.mv(f[tip[1]].r,tip[2])) end
function K.tips(q)
    local f=K.fk(q)
    local a,b=K.point(f,Robot.fixed_tip),K.point(f,Robot.moving_tip)
    return a,b,M.scale(M.add(a,b),0.5),f
end
function K.ik(target,q,roll,iterations)
    local out=table.clone(q)
    if roll then out[5]=math.clamp(roll,Robot.range[5][1],Robot.range[5][2]) end
    for _=1,iterations or 8 do
        local _,_,p,f=K.tips(out)
        local error=M.sub(target,p)
        if M.norm(error)<0.0003 then break end
        local cols={}
        -- Differential midpoint Jacobian; gripper opening is commanded separately.
        for _,b in ipairs(Robot.bodies) do
            if b.joint>0 and b.joint<5 then
                local frame=f[b.joint+1]
                cols[b.joint]=M.cross(frame.axis,M.sub(p,frame.joint))
            end
        end
        local down=M.mv(f[Robot.fixed_tip[1]].r,{0,0,-1})
        local orientation=M.sub({0,0,-1},down)
        local angular={}
        for i=1,4 do angular[i]=M.cross(f[i+1].axis,down) end
        local a,b={},{}
        for i=1,4 do
            a[i]={}; b[i]=M.dot(cols[i],error)+.0025*M.dot(angular[i],orientation)
            for j=1,4 do a[i][j]=M.dot(cols[i],cols[j])+.0025*M.dot(angular[i],angular[j])+(if i==j then .000025 else 0) end
        end
        local e=M.solve(a,b)
        for j=1,4 do out[j]=math.clamp(out[j]+math.clamp(e[j],-0.16,0.16),Robot.range[j][1],Robot.range[j][2]) end
    end
    return out
end
return K
