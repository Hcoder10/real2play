local player=game.Players.LocalPlayer
local remote=game.ReplicatedStorage:WaitForChild("PolishInspect")
local function visible(item)
    while item and item~=player.PlayerGui do
        if item:IsA("GuiObject") and not item.Visible then return false end
        item=item.Parent
    end
    return true
end
remote.OnClientInvoke=function(action)
    local gui=player.PlayerGui:WaitForChild("Real2PlayUI")
    if action=="menu" then gui.Menu.Visible=true end
    if action=="scores" then gui.Leaderboard.Visible=true end
    if action=="close" then gui.Menu.Visible=false; gui.Leaderboard.Visible=false end
    task.wait(.2)
    local rects={}
    for _,o in ipairs(gui:GetDescendants()) do
        if o:IsA("GuiObject") and visible(o) and (o:IsA("TextLabel") or o:IsA("TextButton")) and o.Text~="" then
            local p,s=o.AbsolutePosition,o.AbsoluteSize
            local ancestor=o.Parent; local clipped=false
            while ancestor and ancestor~=gui do
                if ancestor:IsA("ScrollingFrame") then
                    local ap,az=ancestor.AbsolutePosition,ancestor.AbsoluteSize
                    if p.Y+s.Y<=ap.Y or p.Y>=ap.Y+az.Y then clipped=true end
                    local bottom=math.min(p.Y+s.Y,ap.Y+az.Y); p=Vector2.new(p.X,math.max(p.Y,ap.Y)); s=Vector2.new(s.X,math.max(0,bottom-p.Y))
                end
                ancestor=ancestor.Parent
            end
            if not clipped then table.insert(rects,{name=o.Name,text=o.Text,x=p.X,y=p.Y,w=s.X,h=s.Y}) end
        end
    end
    local overlaps={}
    for i,a in ipairs(rects) do for j=i+1,#rects do local b=rects[j]
        if math.min(a.x+a.w,b.x+b.w)-math.max(a.x,b.x)>1 and math.min(a.y+a.h,b.y+b.h)-math.max(a.y,b.y)>1 then
            table.insert(overlaps,a.text.." / "..b.text)
        end
    end end
    local prompts={}
    for _,o in ipairs(workspace.Real2Play:GetDescendants()) do if o:IsA("ProximityPrompt") and o.Enabled then table.insert(prompts,o:GetFullName()) end end
    return {viewport={workspace.CurrentCamera.ViewportSize.X,workspace.CurrentCamera.ViewportSize.Y},overlaps=overlaps,prompts=prompts,rects=rects}
end
