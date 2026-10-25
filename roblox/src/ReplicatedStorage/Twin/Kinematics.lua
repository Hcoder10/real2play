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
    local count=if roll then 4 else 5
    for _=1,iterations or 12 do
        local tipA,tipB,p,f=K.tips(out)
        local error=M.sub(target,p)
        local linear,angular={},{}
        for j=1,count do
            local frame=f[j+1]
            linear[j]=M.cross(frame.axis,M.sub(p,frame.joint)); angular[j]=frame.axis
        end
        local function gram(cols,lambda)
            local a={{lambda,0,0},{0,lambda,0},{0,0,lambda}}
            for _,v in ipairs(cols) do for i=1,3 do for j=1,3 do a[i][j]+=v[i]*v[j] end end end
            return a
        end
        local function inverse(cols,a,v)
            local solved=M.solve(a,v); local out={}
            for j,c in ipairs(cols) do out[j]=M.dot(c,solved) end
            return out
        end
        local across=M.unit(M.sub(tipB,tipA))
        local flatAcross=M.unit({across[1],across[2],0})
        local down=M.mv(f[Robot.fixed_tip[1]].r,{0,0,-1})
        local flatDown=M.unit(M.sub(down,M.scale(across,M.dot(down,across))))
        local orientation=M.add(M.cross(across,flatAcross),M.cross(flatDown,{0,0,-1}))
        local translationGram=gram(linear,.000015)
        local positionStep=inverse(linear,translationGram,{math.clamp(error[1],-.025,.025),math.clamp(error[2],-.025,.025),math.clamp(error[3],-.025,.025)})
        local postureStep=inverse(angular,gram(angular,.01),orientation)
        local shift={0,0,0}
        for j=1,count do shift=M.add(shift,M.scale(linear[j],postureStep[j])) end
        local projection=inverse(linear,translationGram,shift)
        for j=1,count do
            local delta=positionStep[j]+.6*(postureStep[j]-projection[j])
            out[j]=math.clamp(out[j]+math.clamp(delta,-.15,.15),Robot.range[j][1]+1e-5,Robot.range[j][2]-1e-5)
        end
    end
    return out
end
return K
