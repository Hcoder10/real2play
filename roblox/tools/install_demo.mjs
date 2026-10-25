import {readFileSync} from 'node:fs';
import {run,quote,close} from './bridge.mjs';
try {
  if (process.argv.includes('--remove')) {
    await run('local s=game.ServerScriptService:FindFirstChild("Y2Demo"); if s then s:Destroy() end; local c=game.StarterPlayer.StarterPlayerScripts:FindFirstChild("Y2DemoClient"); if c then c:Destroy() end; workspace:SetAttribute("R2PTest",nil); return true');
  } else {
  for (const [service,name,type,path] of [
    ['ServerScriptService','Y2Demo','Script','roblox/tools/polish_demo.luau'],
    ['StarterPlayer','Y2DemoClient','LocalScript','roblox/tools/polish_demo.luau.client.lua']]) {
    await run(`local p=game:GetService(${quote(service)}); if p.Name=="StarterPlayer" then p=p.StarterPlayerScripts end; local old=p:FindFirstChild(${quote(name)}); if old then old:Destroy() end; local s=Instance.new(${quote(type)}); s.Name=${quote(name)}; s.Source=${quote(readFileSync(path,'utf8'))}; s.Parent=p; workspace:SetAttribute("R2PTest",true); return true`);
  }
  }
} finally {await close();}
