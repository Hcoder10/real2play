local G=require(game:GetService("ServerScriptService").ContactGrasp)
local Robot=require(game:GetService("ReplicatedStorage").Twin.Generated.Robot)
local model=Instance.new("Model"); model.Name="ContactGraspTest"; model.Parent=workspace
local function part(name,x,size,anchored)
    local p=Instance.new("Part"); p.Name=name; p.Size=size; p.CFrame=CFrame.new(x,30,0); p.Anchored=anchored; p.Parent=model; return p
end
local fixed=part("FixedFinger",-.4,Vector3.new(.06,.12,.2),true)
local moving=part("MovingFinger",.4,Vector3.new(.06,.12,.2),true)
local cap=part("Cap",0,Vector3.new(.17,.33,.33),false); cap.Shape=Enum.PartType.Cylinder; cap.CFrame*=CFrame.Angles(0,0,math.pi/2)
local arm={model=model,fingerSides={[fixed]="fixed",[moving]="moving"},roots={[Robot.fixed_tip[1]]=fixed},measured=function() return {0,0,0,0,0,.35} end}
local grip=G.new(arm); local roots={piece=cap}; local definitions={{id="piece",role="manipulated"}}
-- A cap centred between open jaws must never be acquired by proximity.
assert(grip:update(true,roots,definitions)==nil and grip:update(true,roots,definitions)==nil)
fixed.Position=Vector3.new(-.17,30,0)
assert(grip:update(true,roots,definitions)==nil and grip:update(true,roots,definitions)==nil,"one-finger contact is not a grasp")
moving.Position=Vector3.new(.17,30,0)
local before=cap.CFrame
assert(grip:update(true,roots,definitions)==nil,"require persistent bilateral contact")
assert(grip:update(true,roots,definitions)==.35 and grip.held,"opposing physical contacts should acquire")
assert(cap.CFrame==before,"acquisition teleported the cap")
assert(grip.held.weld.Part0==fixed and grip.held.weld.Part1==cap,"grasp must be in the fixed gripper frame")
assert(grip:update(false,roots,definitions)==nil and not grip.held,"opening did not release")
assert(cap.CFrame==before,"release teleported the cap")
model:Destroy()
return {pass=true,proximity_rejected=true,single_contact_rejected=true,bilateral_contact=true,pose_preserved=true,release=true}
