-- Pure Luau task/board/camera helper for place_on_cell.
local Tasks = require(if script then script.Parent.Parent.Generated.Tasks else "../Generated/Tasks")
local Game = require(if script then script.Parent.Parent.Generated.Game else "../Generated/Game")
local M = require(if script then script.Parent.Parent.Math3 else "../Math3")

local P = {}
local taskDef = Tasks.tasks.place_on_cell
local boardDef = taskDef.board

P.task_name = taskDef.name
P.instruction = taskDef.instruction
P.round_s = Game.round_s
P.move_s = Game.move_s
P.tutorial = Game.tutorial
P.tutorial_s = Game.tutorial_s
P.board = {
    center_xy = {boardDef.center_xy[1], boardDef.center_xy[2]},
    cell_m = boardDef.cell_m,
    cells = boardDef.cells,
}

P.camera = {
    schema = "r2p_camera/1",
    frame = "spec",
    K = {
        {752.1153743727194, 0.0, 478.74240365741974},
        {0.0, 749.5116465646025, 264.81018854683447},
        {0.0, 0.0, 1.0},
    },
    dist = {0.04317471748937133, -0.06404805243942509, 0.0003388866910065102, -0.0016967964808593662, 0.0},
    R = {
        {0.11134043580394815, 0.9935649319868023, 0.020785410293000628},
        {0.5115364191880085, -0.039366663928370516, -0.8583593406117633},
        {-0.852017487613545, 0.10620259741273819, -0.5126277490572382},
    },
    t = {-0.04038951406252864, -0.007075785273998356, 0.5928439083386667},
    image_size = {960, 540},
}

function P.project(xyz)
    local r = P.camera.R
    local t = P.camera.t
    local k = P.camera.K
    local d = P.camera.dist
    local xc = r[1][1]*xyz[1] + r[1][2]*xyz[2] + r[1][3]*xyz[3] + t[1]
    local yc = r[2][1]*xyz[1] + r[2][2]*xyz[2] + r[2][3]*xyz[3] + t[2]
    local zc = r[3][1]*xyz[1] + r[3][2]*xyz[2] + r[3][3]*xyz[3] + t[3]
    assert(zc > 1e-6, "point behind camera")
    local x = xc / zc
    local y = yc / zc
    local r2 = x*x + y*y
    local radial = 1 + d[1]*r2 + d[2]*r2*r2 + d[5]*r2*r2*r2
    local xd = x*radial + 2*d[3]*x*y + d[4]*(r2 + 2*x*x)
    local yd = y*radial + d[3]*(r2 + 2*y*y) + 2*d[4]*x*y
    return {k[1][1]*xd + k[1][3], k[2][2]*yd + k[2][3], zc}
end

function P.cameraPositionSpec()
    local r = P.camera.R
    local t = P.camera.t
    return {
        -(r[1][1]*t[1] + r[2][1]*t[2] + r[3][1]*t[3]),
        -(r[1][2]*t[1] + r[2][2]*t[2] + r[3][2]*t[3]),
        -(r[1][3]*t[1] + r[2][3]*t[2] + r[3][3]*t[3]),
    }
end

function P.nearestCell(xy)
    local bestId, bestDist = nil, math.huge
    for id, c in pairs(P.board.cells) do
        local dx = xy[1] - c[1]
        local dy = xy[2] - c[2]
        local dist = math.sqrt(dx*dx + dy*dy)
        if dist < bestDist then
            bestId, bestDist = id, dist
        end
    end
    return bestId, bestDist
end

return P
