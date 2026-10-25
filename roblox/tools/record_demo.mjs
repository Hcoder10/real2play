// Record the actual visible Studio Play window. Run after install_demo.mjs + F5.
// FFMPEG and STUDIO_WINDOW_TITLE are explicit; no hardware/GPU encoder is used.
import {spawn} from 'node:child_process';
import {mkdirSync,writeFileSync} from 'node:fs';
import {callTool,close} from './bridge.mjs';
process.env.BRIDGE='server';
const ffmpeg=process.env.FFMPEG;
const title=process.env.STUDIO_WINDOW_TITLE;
if (!ffmpeg || !title) throw new Error('Set FFMPEG and STUDIO_WINDOW_TITLE');
mkdirSync('out/demo',{recursive:true});
const events=[];
async function stage() {
  try {return (await callTool('instance_get_property',{path:'ReplicatedStorage.DemoStage',property:'Value'})).value;} catch {await close(); return '';}
}
async function waitFor(wanted) {
  const deadline=Date.now()+120000;
  while (Date.now()<deadline) {
    const value=await stage();
    if(wanted.includes(value)) return value;
    await new Promise(r=>setTimeout(r,250));
  }
  throw new Error('Timed out waiting for '+wanted);
}
async function capture(name,seconds) {
  const file=`out/demo/roblox_${name}.mp4`;
  events.push({name,start_utc:new Date().toISOString(),stage:await stage(),seconds,file});
  await new Promise((resolve,reject)=>{
    const child=spawn(ffmpeg,['-hide_banner','-loglevel','error','-f','gdigrab','-framerate','30','-i',`title=${title}`,
      '-t',String(seconds),'-vf','scale=1920:-2','-c:v','libx264','-threads','2','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart','-y',file],{stdio:'inherit',windowsHide:true});
    child.on('error',reject); child.on('exit',code=>code===0?resolve():reject(new Error(`ffmpeg exited ${code}`)));
  });
  console.log(file);
}
try {
  if (process.argv.includes('--cap-only')) {
    await waitFor(['cap']); await capture('cap',25);
    writeFileSync('roblox/out/y2_cap_recording.json',JSON.stringify({source:'Roblox Studio Play / ffmpeg gdigrab',events},null,2));
  } else {
  await waitFor(['lobby']); await capture('lobby',20);
  await waitFor(['cap']); await capture('cap',25);
  await waitFor(['bowl']); await capture('bowl',20);
  const report=await callTool('instance_get_property',{path:'ReplicatedStorage.DemoReport',property:'Value'});
  writeFileSync('roblox/out/y2_ui_samples.json',report.value);
  writeFileSync('roblox/out/y2_recording.json',JSON.stringify({source:'Roblox Studio Play / ffmpeg gdigrab',events},null,2));
  }
} finally {await close();}
