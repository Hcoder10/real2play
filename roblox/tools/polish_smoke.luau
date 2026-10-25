-- Run through runmode.mjs; verifies generated geometry and prompt anchors.
while not _G.R2P_HARNESS do task.wait(.1) end
local H=_G.R2P_HARNESS
local A=require(game.ReplicatedStorage.Twin.ArmBuilder)
local checks=0
local function check(value,message) assert(value,message); checks+=1 end
check(H.world.WorkshopFloor.Color==Color3.fromRGB(35,74,82),"floor palette")
for _,s in ipairs(H.stations) do
    check(s.prompt.Parent:IsA("Attachment") and s.watchPrompt.Parent:IsA("Attachment"),"distinct prompt attachments")
    check((s.prompt.Parent.WorldPosition-s.watchPrompt.Parent.WorldPosition).Magnitude>3,"prompt anchor separation")
    check(s.prompt.UIOffset.Y==64 and s.watchPrompt.UIOffset.Y==-64,"prompt UI separation")
    for _,p in ipairs(s.arm.model:GetChildren()) do
        if p.Name=="Visual" then check(p.Color.R<.23 and p.Color.G<.24 and p.Color.B<.25,"black SO-101 arm") end
    end
    if s.index%5~=0 then
        check(s.objects:FindFirstChild("PrintedPaper")~=nil,"white printed paper")
        local lines=0; for _,p in ipairs(s.objects:GetChildren()) do if p.Name=="PrintedGrid" then lines+=1 end end
        check(lines==4,"two lines in each grid direction")
        local cap=s.roots.piece
        check(cap.Shape==Enum.PartType.Cylinder,"cap cylinder")
        check((cap.Size-Vector3.new(.17,.33,.33)).Magnitude<1e-5,"33 x 17 mm cap at ten studs/metre")
        local near=A.unvector(s.objects.WeightNear.Position-s.origin)
        check(math.abs(near[1]-.095)<1e-5 and math.abs(near[2]+.012)<1e-5,"Argon near weight position")
        local far=A.unvector(s.objects.WeightFar.Position-s.origin)
        check(math.abs(far[1]-.215)<1e-5 and math.abs(far[2]+.025)<1e-5,"Argon far weight position")
    else
        check(s.roots.bowl~=nil and s.roots.cube_a~=nil and s.roots.cube_b~=nil,"measured bowl level")
        check(s.roots.bowl.Color.B>s.roots.bowl.Color.R,"blue bowl")
        check(s.roots.cube_a.Color.R>s.roots.cube_a.Color.B and s.roots.cube_b.Color.R>s.roots.cube_b.Color.B,"two brown boxes")
    end
end
return {pass=true,checks=checks,tables=#H.stations,weights="visual bench props; non-colliding"}
