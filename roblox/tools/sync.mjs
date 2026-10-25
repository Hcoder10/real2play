// Sync only this project's managed scripts and asset data into the existing window.
import { readFileSync, readdirSync } from 'node:fs';
import { dirname, join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { run, quote, close } from './bridge.mjs';
const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const sourceRoot = join(root, 'src');
const files = [];
function visit(dir) { for (const item of readdirSync(dir, { withFileTypes: true })) { const path = join(dir, item.name); if (item.isDirectory()) visit(path); else if (item.name.endsWith('.lua')) files.push(path); } }
visit(sourceRoot);
try {
  const running = await run('return game:GetService("RunService"):IsRunning()');
  if (running === true) throw new Error('Stop the current playtest before syncing');
  for (const path of files) {
    const bits = relative(sourceRoot, path).split(sep);
    if (bits[0] === 'StarterPlayerScripts') bits.unshift('StarterPlayer');
    const file = bits.pop();
    const type = file.endsWith('.server.lua') ? 'Script' : file.endsWith('.client.lua') ? 'LocalScript' : 'ModuleScript';
    const name = file.replace(/\.(server|client)\.lua$|\.lua$/g, '');
    const code = `local p=game:GetService(${quote(bits.shift())})\n` + bits.map(bit => `local f=p:FindFirstChild(${quote(bit)}); if not f then f=Instance.new("Folder"); f.Name=${quote(bit)}; f.Parent=p end; p=f\n`).join('') +
      `local s=p:FindFirstChild(${quote(name)}); if s and not s:IsA(${quote(type)}) then s:Destroy(); s=nil end\n` +
      `if not s then s=Instance.new(${quote(type)}); s.Name=${quote(name)}; s.Parent=p end\ns.Source=${quote(readFileSync(path, 'utf8'))}; s:SetAttribute("R2PManaged",true)\nreturn s:GetFullName()`;
    await run(code);
  }
  const assets = JSON.parse(readFileSync(join(root, 'meshes/SO101Meshes.model.json'), 'utf8'));
  await run(`local rs=game:GetService("ReplicatedStorage"); local folder=rs:FindFirstChild("SO101Meshes"); if not folder then folder=Instance.new("Folder"); folder.Name="SO101Meshes"; folder.Parent=rs end\n` + assets.children.map(asset => {
    const p = asset.properties;
    return `do local mesh=folder:FindFirstChild(${quote(asset.name)}); if not mesh then mesh=Instance.new("MeshPart"); mesh.Name=${quote(asset.name)}; mesh.MeshId=${quote(p.MeshId)}; mesh.Size=Vector3.new(${p.Size.join(',')}); mesh.Anchored=true; mesh.CanCollide=false; mesh.Parent=folder end end`;
  }).join('\n') + '\nreturn #folder:GetChildren()');
  console.log(`Synced ${files.length} scripts and ${assets.children.length} mesh assets`);
} finally { await close(); }
