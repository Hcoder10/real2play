local Players=game:GetService("Players")
local remote=game:GetService("ReplicatedStorage"):WaitForChild("R2PSmokeInspect")
remote.OnClientInvoke=function(action)
    if action=="stand" then game:GetService("ReplicatedStorage").Twin.Action:FireServer("stand") end
    local gui=Players.LocalPlayer.PlayerGui:FindFirstChild("Real2PlayUI")
    return {gui=gui~=nil,locked=workspace.CurrentCamera.CameraType==Enum.CameraType.Scriptable,
        controls=gui and gui:FindFirstChild("Controls") and gui.Controls.Visible or false,
        camera_restored=workspace.CurrentCamera.CameraType==Enum.CameraType.Custom,
        table_score=gui and gui:FindFirstChild("TableScore") and gui.TableScore.Text or ""}
end
