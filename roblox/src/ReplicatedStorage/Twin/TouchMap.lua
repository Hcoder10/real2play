local T={}
function T.stick(x,y,deadzone)
    deadzone=deadzone or .12
    if type(x)~="number" or type(y)~="number" or x~=x or y~=y then return 0,0 end
    local mag=math.sqrt(x*x+y*y)
    if mag<=deadzone then return 0,0 end
    local amount=math.min(1,(mag-deadzone)/(1-deadzone))
    return x/mag*amount,y/mag*amount
end
function T.map(x,y,up,roll,look)
    x,y=T.stick(x,y)
    local fx,fy=look[1],look[2]; local n=math.sqrt(fx*fx+fy*fy)
    if n<1e-8 then fx,fy,n=1,0,1 end
    fx/=n; fy/=n
    return {move={fx*y+fy*x,fy*y-fx*x,math.clamp(up,-1,1)},roll=math.clamp(roll,-1,1)}
end
function T.sanitize(packet)
    if type(packet)~="table" or type(packet.move)~="table" then return nil end
    local out={move={},roll=0,fine=packet.fine==true}
    for i=1,3 do
        local n=packet.move[i]; if type(n)~="number" or n~=n or math.abs(n)==math.huge then return nil end
        out.move[i]=math.clamp(n,-1,1)
    end
    local n=packet.roll or 0; if type(n)~="number" or n~=n or math.abs(n)==math.huge then return nil end
    out.roll=math.clamp(n,-1,1)
    if type(packet.grip)=="boolean" then out.grip=packet.grip end
    return out
end
return T
