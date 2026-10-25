local L=require("../../roblox/src/ReplicatedStorage/Twin/Levels")
local C=require("../../roblox/src/ReplicatedStorage/Twin/TaskCheck")
local T=require("../../roblox/src/ReplicatedStorage/Twin/TicTacToe")
local Managed=require("../../roblox/src/ReplicatedStorage/Twin/Managed/PlaceOnCell")
local G=require("../../roblox/src/ReplicatedStorage/Twin/Generated/Game")
assert(G.tables==20 and G.max_active==20)
assert(T.pieceSize[1]==.033 and T.pieceSize[2]==.033 and T.pieceSize[3]==.017,"cylinder must expose full XYZ extents")
assert(Managed.nearestCell({.1525,.0068})=="r1c1")
assert(Managed.board.cell_m==.0368)
local task=L.tasks.place_on_cell
assert(task.bench_calibration.xy_offset_mm[1]==10 and task.bench_calibration.xy_offset_mm[2]==11)
local n=0
for seed=1,100 do
    local scene=L.sample(task,seed)
    for _,o in ipairs(scene.objects) do
        if o.id=="piece" then
            assert(o.shape=="cylinder" and o.size_m[1]==.033 and o.size_m[2]==.017)
            assert((o.xyz[1]==.142 and o.xyz[2]==-.068) or (o.xyz[1]==.218 and o.xyz[2]==-.075),"supply must use measured off-board cap positions")
            assert(math.abs(o.xyz[3]-.0085)<1e-10)
        end
        local cell=task.board.cells[o.id]
        if cell then assert(o.xyz[1]==cell[1] and o.xyz[2]==cell[2]); assert(o.yaw_deg< -12 and o.yaw_deg> -13) end
        n+=1
    end
end
print("measured bench: "..n.." object checks; caps, rotated board, supply and managed helper passed")
