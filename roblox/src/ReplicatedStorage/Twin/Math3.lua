--!native
-- Small allocation-conscious rigid-body math; pure Luau, SI units.
local M = {}
function M.add(a,b) return {a[1]+b[1],a[2]+b[2],a[3]+b[3]} end
function M.sub(a,b) return {a[1]-b[1],a[2]-b[2],a[3]-b[3]} end
function M.scale(a,s) return {a[1]*s,a[2]*s,a[3]*s} end
function M.dot(a,b) return a[1]*b[1]+a[2]*b[2]+a[3]*b[3] end
function M.cross(a,b) return {a[2]*b[3]-a[3]*b[2],a[3]*b[1]-a[1]*b[3],a[1]*b[2]-a[2]*b[1]} end
function M.norm(a) return math.sqrt(M.dot(a,a)) end
function M.unit(a) return M.scale(a,1/math.max(1e-12,M.norm(a))) end
function M.eye() return {1,0,0,0,1,0,0,0,1} end
function M.mv(a,b) return {a[1]*b[1]+a[2]*b[2]+a[3]*b[3],a[4]*b[1]+a[5]*b[2]+a[6]*b[3],a[7]*b[1]+a[8]*b[2]+a[9]*b[3]} end
function M.transpose(a) return {a[1],a[4],a[7],a[2],a[5],a[8],a[3],a[6],a[9]} end
function M.mm(a,b)
    local c=table.create(9,0)
    for i=0,2 do for j=1,3 do c[i*3+j]=a[i*3+1]*b[j]+a[i*3+2]*b[j+3]+a[i*3+3]*b[j+6] end end
    return c
end
function M.rotation(axis, angle)
    local x,y,z=axis[1],axis[2],axis[3]
    local c,s=math.cos(angle),math.sin(angle); local v=1-c
    return {x*x*v+c,x*y*v-z*s,x*z*v+y*s,y*x*v+z*s,y*y*v+c,y*z*v-x*s,z*x*v-y*s,z*y*v+x*s,z*z*v+c}
end
function M.solve(a,b)
    local n=#b; local rows={}; local out=table.clone(b)
    for i=1,n do rows[i]=table.clone(a[i]) end
    for i=1,n do
        local p=i
        for j=i+1,n do if math.abs(rows[j][i])>math.abs(rows[p][i]) then p=j end end
        assert(math.abs(rows[p][i])>1e-14,"singular dynamics matrix")
        rows[i],rows[p]=rows[p],rows[i]; out[i],out[p]=out[p],out[i]
        for j=i+1,n do
            local f=rows[j][i]/rows[i][i]
            for k=i+1,n do rows[j][k]-=f*rows[i][k] end
            out[j]-=f*out[i]
        end
    end
    for i=n,1,-1 do
        for j=i+1,n do out[i]-=rows[i][j]*out[j] end
        out[i]/=rows[i][i]
    end
    return out
end
return M
