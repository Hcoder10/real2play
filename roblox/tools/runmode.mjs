// Execute a test in the real Run/Play server, using StudioTestService's test protocol.
import { readFileSync, writeFileSync } from 'node:fs';
import { run, quote, close } from './bridge.mjs';
const [file, arg = '{}', timeout = '180', output] = process.argv.slice(2);
const json = (arg.startsWith('@') ? readFileSync(arg.slice(1), 'utf8') : arg).replace(/^\uFEFF/, '');
JSON.parse(json); // Reject invalid command-line JSON before starting Studio.
const source = `local service=game:GetService("StudioTestService")\nlocal ok,result=pcall(function()\n_G.R2P_ARGS=game:GetService("HttpService"):JSONDecode(service:GetTestArgs().json)\n${readFileSync(file,'utf8')}\nend)\nservice:EndTest({ok=ok,value=ok and result or nil,error=not ok and tostring(result) or nil})`;
try {
  await run(`local scripts=game:GetService("ServerScriptService"); local old=scripts:FindFirstChild("R2PTestJob"); if old then old:Destroy() end\nlocal job=Instance.new("Script"); job.Name="R2PTestJob"; job.Source=${quote(source)}; job.Parent=scripts\nworkspace:SetAttribute("R2PTest",true); _G.R2P_TEST_DONE=false; _G.R2P_TEST_RESULT=nil\ntask.spawn(function() local ok,result=pcall(function() return game:GetService("StudioTestService"):${process.env.PLAY ? 'ExecutePlayModeAsync' : 'ExecuteRunModeAsync'}({json=${quote(json)}}) end); _G.R2P_TEST_RESULT=ok and result or {ok=false,error=tostring(result)}; _G.R2P_TEST_DONE=true end)\nreturn "started"`);
  const deadline = Date.now() + Number(timeout)*1000;
  let result;
  while (Date.now() < deadline) {
    await new Promise(resolve=>setTimeout(resolve,1200));
    try { const poll=await run('return {done=_G.R2P_TEST_DONE,result=_G.R2P_TEST_RESULT}'); if(poll?.done){result=poll.result;break;} } catch {}
  }
  if (!result) throw new Error('Studio test timed out');
  if (output) writeFileSync(output,JSON.stringify(result,null,2));
  console.log(JSON.stringify(result,null,2));
  if (!result.ok) process.exitCode=1;
} catch(error) { console.error(error.message);process.exitCode=1; }
finally {
  await run('local s=game:GetService("ServerScriptService"):FindFirstChild("R2PTestJob"); if s then s:Destroy() end; workspace:SetAttribute("R2PTest",nil); return true').catch(()=>{});
  await close();
}
