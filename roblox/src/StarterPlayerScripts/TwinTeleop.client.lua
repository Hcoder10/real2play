local Players=game:GetService("Players")
local Run=game:GetService("RunService")
local Input=game:GetService("UserInputService")
local Actions=game:GetService("ContextActionService")
local Tween=game:GetService("TweenService")
local Haptic=game:GetService("HapticService")
local Twin=game:GetService("ReplicatedStorage"):WaitForChild("Twin")
local Touch=require(Twin.TouchMap)
local Levels=require(Twin.Levels)
local Game=require(Twin.Generated.Game)
local inputRemote=Twin:WaitForChild("Input")
local stateRemote=Twin:WaitForChild("State")
local actionRemote=Twin:WaitForChild("Action")
local player=Players.LocalPlayer
local camera=workspace.CurrentCamera
local lobbyFieldOfView=camera.FieldOfView
local gui=Instance.new("ScreenGui"); gui.Name="Real2PlayUI"; gui.ResetOnSpawn=false; gui.IgnoreGuiInset=false; gui.ZIndexBehavior=Enum.ZIndexBehavior.Sibling; gui.Parent=player:WaitForChild("PlayerGui")
local navy=Color3.fromRGB(unpack(Game.theme.navy)); local blue=Color3.fromRGB(unpack(Game.theme.blue)); local paper=Color3.fromRGB(unpack(Game.theme.paper)); local yellow=Color3.fromRGB(unpack(Game.theme.yellow))
local function corner(obj,r) local c=Instance.new("UICorner"); c.CornerRadius=UDim.new(0,r or 12); c.Parent=obj end
local function label(parent,text,size,pos,dims)
    local l=Instance.new("TextLabel"); l.Text=text; l.TextSize=size; l.Font=Enum.Font.GothamMedium; l.TextColor3=navy; l.BackgroundTransparency=1
    l.Position=pos; l.Size=dims; l.TextXAlignment=Enum.TextXAlignment.Left; l.TextWrapped=true; l.Parent=parent; return l
end
local function button(parent,text,pos,dims,callback,fill)
    local b=Instance.new("TextButton"); b.Text=text; b.Font=Enum.Font.GothamBold; b.TextSize=17; b.TextColor3=Color3.new(1,1,1)
    b.BackgroundColor3=fill or blue; b.Position=pos; b.Size=dims; b.AutoButtonColor=true; b.Parent=parent; corner(b,12)
    if callback then b.Activated:Connect(callback) end
    return b
end
local bar=Instance.new("Frame"); bar.Size=UDim2.new(1,-24,0,66); bar.Position=UDim2.fromOffset(12,8); bar.BackgroundColor3=paper; bar.Parent=gui; corner(bar,16)
local brand=label(bar,"Teach the robot",24,UDim2.fromOffset(16,4),UDim2.new(.6,0,0,32)); brand.Font=Enum.Font.FredokaOne
local counter=label(bar,"Take a seat at any blue chair",13,UDim2.fromOffset(16,37),UDim2.new(1,-145,0,22))
local timer=label(gui,"",25,UDim2.new(.5,-40,0,80),UDim2.fromOffset(80,34)); timer.TextXAlignment=Enum.TextXAlignment.Center
local menu=Instance.new("Frame"); menu.Name="Menu"; menu.Size=UDim2.fromOffset(360,480); menu.Position=UDim2.fromOffset(16,88); menu.BackgroundColor3=paper; menu.Parent=gui; corner(menu,18)
local headline=label(menu,"Your hands.\nA robot’s next skill.",31,UDim2.fromOffset(20,15),UDim2.new(1,-40,0,82)); headline.Font=Enum.Font.FredokaOne
local instruction=label(menu,"Pick up a piece, play a round, and help a real robot learn.",15,UDim2.fromOffset(20,100),UDim2.new(1,-40,0,45))
local seated,active,preview=false,false,false
local stationId,origin,view=nil,nil,nil
local selectedLevel=Levels.order[1]; local selectedMode="solo"
local closed=false; local stats={rounds=0,sent=0,stars=0,points=0}; local tutorialEnd=0; local shake=0
local function restoreCamera()
    preview=false; camera.CameraType=Enum.CameraType.Custom; camera.FieldOfView=lobbyFieldOfView
    local h=player.Character and player.Character:FindFirstChildOfClass("Humanoid"); if h then camera.CameraSubject=h end
end
local function play(level,mode)
    if not seated then instruction.Text="Walk to a blue chair, then choose Sit & play."; menu.Visible=false; restoreCamera(); return end
    selectedLevel=level or selectedLevel; selectedMode=mode or selectedMode
    actionRemote:FireServer("play",{level=selectedLevel,mode=selectedMode})
end
local main=button(menu,"Play the robot",UDim2.fromOffset(20,156),UDim2.new(1,-40,0,58),function() play("place_on_cell","tic_tac_toe") end)
local section=label(menu,"Or teach a skill",16,UDim2.fromOffset(20,227),UDim2.new(1,-40,0,25)); section.Font=Enum.Font.GothamBold
local list=Instance.new("ScrollingFrame"); list.Position=UDim2.fromOffset(20,260); list.Size=UDim2.new(1,-40,0,148); list.BackgroundTransparency=1; list.BorderSizePixel=0; list.ScrollBarThickness=4; list.CanvasSize=UDim2.fromOffset(0,#Levels.order*48); list.Parent=menu
for i,name in ipairs(Levels.order) do
    local title=name:gsub("_"," "); title=title:sub(1,1):upper()..title:sub(2)
    local b=button(list,title,UDim2.fromOffset(0,(i-1)*48),UDim2.new(1,-8,0,42),function() play(name,selectedMode) end,Color3.fromRGB(220,232,242)); b.TextColor3=navy; b.TextSize=14
end
local modes={"solo","vs_bot","timed","daily"}; local modeNames={"Solo","Race bot","Timed","Daily"}
local modeIndex=1
button(menu,"Mode: Solo",UDim2.new(0,20,1,-54),UDim2.new(1,-132,0,38),function()
    modeIndex=modeIndex%#modes+1; selectedMode=modes[modeIndex]
    menu.ModeButton.Text="Mode: "..modeNames[modeIndex]
end).Name="ModeButton"
local colorIndex=0
button(menu,"Color",UDim2.new(1,-104,1,-54),UDim2.fromOffset(84,38),function()
    local available={}; for i,u in ipairs(Game.unlocks) do if (stats.stars or 0)>=u.stars then table.insert(available,i) end end
    if #available==0 then instruction.Text="Earn 2 stars to unlock your first gripper color."; return end
    colorIndex=colorIndex%#available+1; actionRemote:FireServer("color",available[colorIndex])
end,Color3.fromRGB(42,143,118))
local real=Instance.new("Frame"); real.Name="RealArm"; real.Position=UDim2.new(1,-226,0,90); real.Size=UDim2.fromOffset(210,155); real.BackgroundColor3=paper; real.Parent=gui; corner(real,16)
label(real,"Real arm",18,UDim2.fromOffset(12,8),UDim2.new(1,-24,0,24)).Font=Enum.Font.GothamBold
local realCaption=label(real,"Your play helps it learn",11,UDim2.fromOffset(12,128),UDim2.new(1,-24,0,20))
local photo=Instance.new("ImageLabel"); photo.Position=UDim2.fromOffset(10,36); photo.Size=UDim2.new(1,-20,0,88); photo.BackgroundTransparency=1; photo.ScaleType=Enum.ScaleType.Fit; photo.Visible=false; photo.Parent=real
local drawing=Instance.new("Frame"); drawing.Position=photo.Position; drawing.Size=photo.Size; drawing.BackgroundTransparency=1; drawing.Parent=real
local links={}
for i=1,3 do local p=Instance.new("Frame"); p.Size=UDim2.fromOffset(if i==3 then 22 else 62,12); p.AnchorPoint=Vector2.new(0,.5); p.Position=UDim2.fromOffset(30+(i-1)*45,if i==1 then 75 elseif i==2 then 28 else 14); p.Rotation=if i==1 then -65 elseif i==2 then 10 else 60; p.BackgroundColor3=yellow; p.Parent=drawing; corner(p,6); links[i]=p end
local leaderboard=Instance.new("Frame"); leaderboard.Size=UDim2.fromOffset(320,300); leaderboard.AnchorPoint=Vector2.new(.5,.5); leaderboard.Position=UDim2.fromScale(.5,.5); leaderboard.BackgroundColor3=paper; leaderboard.Visible=false; leaderboard.Parent=gui; corner(leaderboard,18)
label(leaderboard,"Top teachers",26,UDim2.fromOffset(20,16),UDim2.new(1,-40,0,40)).Font=Enum.Font.FredokaOne
local leader=label(leaderboard,"Be the first to teach a skill.",16,UDim2.fromOffset(20,65),UDim2.new(1,-40,0,160)); leader.TextYAlignment=Enum.TextYAlignment.Top
button(leaderboard,"Back",UDim2.new(0,20,1,-62),UDim2.new(1,-40,0,42),function() leaderboard.Visible=false end)
local scoresButton=button(menu,"Scores",UDim2.new(1,-104,0,224),UDim2.fromOffset(84,30),function() leaderboard.Visible=true end,navy); scoresButton.TextSize=12
local controls=Instance.new("Frame"); controls.Name="Controls"; controls.Size=UDim2.fromScale(1,1); controls.BackgroundTransparency=1; controls.Visible=false; controls.Parent=gui
local stick=Instance.new("Frame"); stick.Name="MoveStick"; stick.AnchorPoint=Vector2.new(0,1); stick.Position=UDim2.new(0,28,1,-30); stick.Size=UDim2.fromOffset(150,150); stick.BackgroundColor3=paper; stick.BackgroundTransparency=.12; stick.Active=true; stick.Parent=controls; corner(stick,75)
local knob=Instance.new("Frame"); knob.AnchorPoint=Vector2.new(.5,.5); knob.Position=UDim2.fromScale(.5,.5); knob.Size=UDim2.fromOffset(58,58); knob.BackgroundColor3=blue; knob.Parent=stick; corner(knob,29)
label(stick,"Move",14,UDim2.fromOffset(0,-30),UDim2.new(1,0,0,26)).TextXAlignment=Enum.TextXAlignment.Center
local xy={0,0}; local gamepad={0,0}; local keys={}; local holds={}; local stickTouch=nil
local function resetAxes()
    xy={0,0}; gamepad={0,0}; keys={}; holds={}; stickTouch=nil; knob.Position=UDim2.fromScale(.5,.5)
end
Input.WindowFocusReleased:Connect(resetAxes)
local function joystick(position)
    local delta=Vector2.new(position.X,position.Y)-(stick.AbsolutePosition+stick.AbsoluteSize/2)
    local length=delta.Magnitude; if length>58 then delta=delta/length*58 end
    knob.Position=UDim2.new(.5,delta.X,.5,delta.Y); xy={delta.X/58,-delta.Y/58}
end
stick.InputBegan:Connect(function(input) if input.UserInputType==Enum.UserInputType.Touch or input.UserInputType==Enum.UserInputType.MouseButton1 then stickTouch=input; joystick(input.Position) end end)
local function holdButton(text,x,y,key)
    local b=button(controls,text,UDim2.new(1,x,1,y),UDim2.fromOffset(76,58),nil,blue)
    b.InputBegan:Connect(function(input) if input.UserInputType==Enum.UserInputType.Touch or input.UserInputType==Enum.UserInputType.MouseButton1 then holds[input]=key; keys[key]=true end end)
    return b
end
holdButton("Up",-194,-238,"up"); holdButton("Down",-106,-238,"down")
holdButton("↶ Twist",-194,-168,"leftRoll"); holdButton("Twist ↷",-106,-168,"rightRoll")
local grab
grab=button(controls,"Grab",UDim2.new(1,-194,1,-98),UDim2.fromOffset(164,68),function() closed=not closed; grab.Text=if closed then "Drop" else "Grab" end,Color3.fromRGB(42,143,118))
local help=label(controls,"WASD move · Q/E height · Z/X twist · Space grab · Shift slow",12,UDim2.new(.5,-240,1,-24),UDim2.fromOffset(480,22)); help.TextXAlignment=Enum.TextXAlignment.Center
local stand=button(gui,"Stand up",UDim2.new(1,-115,0,18),UDim2.fromOffset(92,42),function() actionRemote:FireServer("stand") end,navy); stand.Visible=false; stand.TextSize=13
local menuButton=button(gui,"Skills",UDim2.new(0,16,0,88),UDim2.fromOffset(88,40),function() menu.Visible=not menu.Visible end,navy); menuButton.Visible=false
local coach=label(gui,"",19,UDim2.new(.5,-210,0,117),UDim2.fromOffset(420,56)); coach.TextXAlignment=Enum.TextXAlignment.Center; coach.Font=Enum.Font.GothamBold
local ghost=label(gui,"☝",46,UDim2.fromOffset(0,0),UDim2.fromOffset(60,60)); ghost.TextColor3=blue; ghost.Visible=false
local resultPanel=Instance.new("Frame"); resultPanel.Size=UDim2.fromOffset(320,230); resultPanel.AnchorPoint=Vector2.new(.5,.5); resultPanel.Position=UDim2.fromScale(.5,.45); resultPanel.BackgroundColor3=paper; resultPanel.Visible=false; resultPanel.Parent=gui; corner(resultPanel,20)
local resultTitle=label(resultPanel,"",30,UDim2.fromOffset(20,20),UDim2.new(1,-40,0,55)); resultTitle.TextXAlignment=Enum.TextXAlignment.Center; resultTitle.Font=Enum.Font.FredokaOne
local resultBody=label(resultPanel,"",17,UDim2.fromOffset(20,80),UDim2.new(1,-40,0,65)); resultBody.TextXAlignment=Enum.TextXAlignment.Center
button(resultPanel,"Play again",UDim2.fromOffset(20,163),UDim2.new(1,-40,0,48),function() resultPanel.Visible=false; play() end)
local function setImage(id) photo.Image=id or ""; photo.Visible=photo.Image~=""; drawing.Visible=not photo.Visible; realCaption.Text=if photo.Visible then "The real robot you’re teaching" else "Animated preview" end
local function juice(kind)
    local soundId=if kind=="win" then Game.juice.win elseif kind=="grab" then Game.juice.grab else Game.juice.drop
    local sound=Instance.new("Sound"); sound.SoundId=soundId; sound.Volume=.28; sound.Parent=gui; sound:Play(); game:GetService("Debris"):AddItem(sound,3)
    shake=if kind=="win" then .5 else .15
    pcall(function()
        local device=if Input.TouchEnabled then Enum.UserInputType.Touch else Enum.UserInputType.Gamepad1
        if Haptic:IsMotorSupported(device,Enum.VibrationMotor.Small) then Haptic:SetMotor(device,Enum.VibrationMotor.Small,Game.juice.haptic); task.delay(.12,function() Haptic:SetMotor(device,Enum.VibrationMotor.Small,0) end) end
    end)
    if kind=="win" then
        for i=1,Game.juice.particles do
            local p=Instance.new("Frame"); p.Size=UDim2.fromOffset(7,13); p.Position=UDim2.fromScale(.5,.35); p.BackgroundColor3=if i%2==0 then yellow else blue; p.Rotation=math.random(0,180); p.Parent=gui
            Tween:Create(p,TweenInfo.new(1.1,Enum.EasingStyle.Quad),{Position=UDim2.fromScale(math.random(),1.2),Rotation=math.random(-360,360),BackgroundTransparency=1}):Play(); game:GetService("Debris"):AddItem(p,1.2)
        end
    end
end
Input.InputChanged:Connect(function(input)
    if stickTouch and (input==stickTouch or (stickTouch.UserInputType==Enum.UserInputType.MouseButton1 and input.UserInputType==Enum.UserInputType.MouseMovement)) then joystick(input.Position) end
    if input.KeyCode==Enum.KeyCode.Thumbstick1 then
        gamepad={input.Position.X,input.Position.Y}
        if preview and input.Position.Magnitude>.2 then restoreCamera(); menu.Visible=false end
    end
    if input.KeyCode==Enum.KeyCode.ButtonR2 then keys.up=input.Position.Z>.15 end
    if input.KeyCode==Enum.KeyCode.ButtonL2 then keys.down=input.Position.Z>.15 end
end)
Input.InputEnded:Connect(function(input)
    if input==stickTouch then stickTouch=nil; xy={0,0}; knob.Position=UDim2.fromScale(.5,.5) end
    if holds[input] then keys[holds[input]]=false; holds[input]=nil end
end)
local mapping={[Enum.KeyCode.W]="forward",[Enum.KeyCode.S]="back",[Enum.KeyCode.A]="left",[Enum.KeyCode.D]="right",[Enum.KeyCode.Q]="up",[Enum.KeyCode.E]="down",[Enum.KeyCode.Z]="leftRoll",[Enum.KeyCode.X]="rightRoll",[Enum.KeyCode.LeftShift]="fine",[Enum.KeyCode.ButtonL1]="leftRoll",[Enum.KeyCode.ButtonR1]="rightRoll",[Enum.KeyCode.ButtonX]="fine"}
local bindings={}; for key in pairs(mapping) do table.insert(bindings,key) end; for _,key in ipairs({Enum.KeyCode.Space,Enum.KeyCode.ButtonA,Enum.KeyCode.ButtonY,Enum.KeyCode.R,Enum.KeyCode.Return}) do table.insert(bindings,key) end
Actions:BindActionAtPriority("RobotControl",function(_,state,input)
    if not seated or Input:GetFocusedTextBox() then
        if preview and mapping[input.KeyCode] and state==Enum.UserInputState.Begin then restoreCamera(); menu.Visible=false end
        return Enum.ContextActionResult.Pass
    end
    if mapping[input.KeyCode] then keys[mapping[input.KeyCode]]=state~=Enum.UserInputState.End and state~=Enum.UserInputState.Cancel
    elseif state==Enum.UserInputState.Begin then
        if input.KeyCode==Enum.KeyCode.Space or input.KeyCode==Enum.KeyCode.ButtonA then if active then closed=not closed; grab.Text=if closed then "Drop" else "Grab" else play() end
        elseif input.KeyCode==Enum.KeyCode.Return or input.KeyCode==Enum.KeyCode.R or input.KeyCode==Enum.KeyCode.ButtonY then play() end
    end
    return Enum.ContextActionResult.Sink
end,false,3000,unpack(bindings))
stateRemote.OnClientEvent:Connect(function(data)
    if data.stats then stats=data.stats end
    if data.image~=nil then setImage(data.image) end
    if data.leaders then
        local rows={}
        for i,entry in ipairs(data.leaders) do table.insert(rows,string.format("%d. %s    %d",i,entry.name or ("Teacher "..entry.key),entry.value)) end
        leader.Text=if #rows>0 then table.concat(rows,"\n\n") else "Be the first to teach a skill.\n\nYour points: "..(stats.points or 0)
    end
    counter.Text=string.format("%d rounds today, you’ve sent %d to the robot   •   %d stars",stats.rounds or 0,stats.sent or 0,stats.stars or 0)
    if data.kind=="lobby" then
        resetAxes()
        seated=false; active=false; controls.Visible=false; stand.Visible=false; menu.Visible=true; menuButton.Visible=false; resultPanel.Visible=false; leaderboard.Visible=false; coach.Text="Walk up and sit at a table to start."; timer.Text=""
        restoreCamera()
        local world=workspace:FindFirstChild("Real2Play"); local tableModel=world and world:FindFirstChild("Table"..Game.tables)
        if tableModel and data.preview~=false then
            local at=tableModel:GetAttribute("Origin")
            if at then preview=true; view=CFrame.lookAt(at+Vector3.new(5.7,5.3,5.7),at+Vector3.new(2,.4,0)); camera.CameraType=Enum.CameraType.Scriptable end
        end
    elseif data.kind=="seated" then
        seated=true; active=false; preview=false; stationId=data.table; origin=Vector3.new(unpack(data.origin)); view=CFrame.lookAt(origin+Vector3.new(5.7,5.3,5.7),origin+Vector3.new(2.0,.4,0))
        camera.CameraType=Enum.CameraType.Scriptable; camera.FieldOfView=48; camera.CFrame=view; stand.Visible=true; menu.Visible=true; menuButton.Visible=true; instruction.Text="Choose a game. Your robot is ready."; coach.Text=""
    elseif data.kind=="start" then
        resetAxes()
        active=true; closed=false; grab.Text="Grab"; controls.Visible=true; menu.Visible=false; resultPanel.Visible=false; selectedMode=data.mode; selectedLevel=data.level
        coach.Text=data.instruction or ""; tutorialEnd=if data.tutorial then os.clock()+Game.tutorial_s else 0
    elseif data.kind=="tick" then
        timer.Text=string.format("%ds",math.ceil(data.remaining or 0)); timer.TextColor3=if (data.remaining or 0)<10 then Color3.fromRGB(210,77,79) else navy
        local second=math.ceil(data.remaining or 0)
        if second<=10 and second>0 and timer:GetAttribute("LastSecond")~=second then
            timer:SetAttribute("LastSecond",second)
            local tick=Instance.new("Sound"); tick.SoundId=Game.juice.grab; tick.Volume=.12; tick.Parent=gui; tick:Play(); game:GetService("Debris"):AddItem(tick,2)
        end
    elseif data.kind=="board" then coach.Text=if data.phase=="robot_thinking" then "Robot is choosing a move…" else "Place your piece on any free square."
    elseif data.kind=="result" then
        active=false; controls.Visible=false; resultPanel.Visible=true; coach.Text=""; ghost.Visible=false
        resultTitle.Text=if data.result==1 then "You beat the robot!" elseif data.result==0 then "A thoughtful draw." elseif data.success then "You taught a skill!" else "One more try?"
        local reward=data.reward or {}; resultBody.Text=if data.success then string.rep("★",reward.stars or 1).."\n+"..(reward.points or 0).." points · Combo ×"..(reward.combo or 1) else (if data.reason=="timeout" then "Time’s up. Try slower, smaller moves." else "Reset the table and try again.")
        if data.mode=="vs_bot" and data.beat_bot~=nil and data.success then
            resultBody.Text..="\n"..(if data.beat_bot then "You finished before the robot." else "The robot finished first. Race again!")
        end
        if data.success then juice("win") end
    elseif data.kind=="juice" then juice(data.effect)
    elseif data.kind=="notice" then coach.Text=data.text
    end
end)
local clock=0
Run.RenderStepped:Connect(function(dt)
    if (seated or preview) and view then
        shake=math.max(0,shake-dt); camera.CFrame=view*CFrame.new((math.random()-.5)*shake*Game.juice.shake,(math.random()-.5)*shake*Game.juice.shake,0)
    end
    if drawing.Visible then links[2].Rotation=10+math.sin(os.clock()*1.4)*12 end
    if active and os.clock()<tutorialEnd then
        local elapsed=Game.tutorial_s-(tutorialEnd-os.clock()); coach.Text=Game.tutorial[math.min(3,1+math.floor(elapsed/(Game.tutorial_s/3)))]
        ghost.Visible=true; ghost.Position=UDim2.new(.5,math.sin(os.clock()*1.7)*70,0,170+math.cos(os.clock())*18)
    else ghost.Visible=false end
    clock+=dt
    if active and clock>=1/30 then
        clock%=1/30
        local function axis(a,b) return (if keys[a] then 1 else 0)-(if keys[b] then 1 else 0) end
        local x=xy[1]+gamepad[1]+axis("right","left"); local y=xy[2]+gamepad[2]+axis("forward","back")
        local look=camera.CFrame.LookVector; local packet=Touch.map(x,y,axis("up","down"),axis("rightRoll","leftRoll"),{look.X,-look.Z})
        packet.grip=closed; packet.fine=keys.fine==true; inputRemote:FireServer(packet)
    end
end)
local function resize()
    local size=camera.ViewportSize; local small=size.X<700
    menu.Size=UDim2.fromOffset(math.min(360,size.X-32),math.min(480,size.Y-110)); menu.Position=UDim2.fromOffset(16,88)
    list.Position=UDim2.fromOffset(20,260); list.Size=UDim2.new(1,-40,0,math.max(48,menu.Size.Y.Offset-332))
    headline.TextSize=31; headline.Size=UDim2.new(1,-40,0,82); instruction.Visible=true; main.Position=UDim2.fromOffset(20,156); section.Position=UDim2.fromOffset(20,227)
    scoresButton.Position=UDim2.new(1,-104,0,224)
    stick.Size=UDim2.fromOffset(if size.X<400 or size.Y<520 then 120 else 150,if size.X<400 or size.Y<520 then 120 else 150)
    real.Visible=not small; help.Visible=not Input.TouchEnabled; brand.TextSize=if small then 20 else 24
    coach.Size=UDim2.fromOffset(math.min(420,size.X-40),56); coach.Position=UDim2.new(.5,-coach.Size.X.Offset/2,0,117); coach.TextSize=if small then 14 else 19
    if size.Y<520 then stick.Size=UDim2.fromOffset(120,120); menu.Position=UDim2.fromOffset(12,80); headline.TextSize=23; headline.Size=UDim2.new(1,-40,0,54); instruction.Visible=false; main.Position=UDim2.fromOffset(20,80); section.Position=UDim2.fromOffset(20,143); scoresButton.Position=UDim2.new(1,-104,0,140); list.Position=UDim2.fromOffset(20,174); list.Size=UDim2.new(1,-40,0,math.max(40,menu.Size.Y.Offset-232)) end
end
camera:GetPropertyChangedSignal("ViewportSize"):Connect(resize); resize()
actionRemote:FireServer("ready")
