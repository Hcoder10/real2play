local Twin=game:GetService("ReplicatedStorage").Twin
local Rules=require(Twin.TicTacToe)
local A=require(Twin.ArmBuilder)
local G=require(Twin.Generated.Game)
local Match={}; Match.__index=Match
function Match.new(station,onResult,onMove)
    local marks=Instance.new("Folder"); marks.Name="BoardMarks"; marks.Parent=station.model
    return setmetatable({station=station,board=Rules.new(),marks=marks,pieces={},turn=0,onResult=onResult,onMove=onMove,alive=true},Match)
end
function Match:startMove()
    if not self.alive or not self.station.owner then return end
    self.turn+=1
    self.station:start("place_on_cell",self.turn+math.floor(os.clock()*100),"human",{tictactoe=true,board=self.board,limit=G.move_s,boardPieces=self.pieces})
    if self.onMove then self.onMove(self,"your_turn") end
end
function Match:mark(cell,player)
    table.insert(self.pieces,{cell=cell,player=player})
    local xy=Rules.xy(cell)
    local size=Rules.pieceSize
    local p=A.part(self.marks,if player==1 then "X" else "O",Vector3.new(size[1],size[3],size[2])*10,CFrame.new(self.station.origin+A.vector({xy[1],xy[2],size[3]/2})),
        if player==1 then Color3.fromRGB(172,116,76) else Color3.fromRGB(47,113,200),true)
    p.CanCollide=false
    p.Shape=Enum.PartType.Cylinder; p.Size=Vector3.new(size[3],size[1],size[2])*10; p.CFrame*=CFrame.Angles(0,0,math.pi/2)
    local surface=Instance.new("SurfaceGui"); surface.Face=Enum.NormalId.Right; surface.CanvasSize=Vector2.new(100,100); surface.Parent=p
    local label=Instance.new("TextLabel"); label.Size=UDim2.fromScale(1,1); label.BackgroundTransparency=1; label.Text=if player==1 then "X" else "O"; label.TextScaled=true; label.Font=Enum.Font.GothamBold; label.TextColor3=Color3.new(1,1,1); label.Parent=surface
end
function Match:accepted(success,reason,extra)
    if not self.alive then return end
    if not success then self.alive=false; self.onResult(self,2,reason); return end
    local cell=extra.target_cell
    assert(Rules.place(self.board,cell,1),"illegal human board transition")
    self:mark(cell,1)
    local result=Rules.result(self.board)
    if result~=nil then self.alive=false; self.onResult(self,result,""); return end
    if self.onMove then self.onMove(self,"robot_thinking") end
    task.delay(.8,function()
        if not self.alive or not self.station.owner then return end
        local move=Rules.choose(self.board,G.robot_slip)
        assert(Rules.place(self.board,move,2),"illegal robot board transition")
        self:mark(move,2)
        local outcome=Rules.result(self.board)
        if outcome~=nil then self.alive=false; self.onResult(self,outcome,"") else self:startMove() end
    end)
end
function Match:destroy() self.alive=false; self.marks:Destroy() end
return Match
