import {writeFileSync} from 'node:fs';
import {callTool,close} from './bridge.mjs';
process.env.BRIDGE='server';
try {
  const stage=await callTool('instance_get_property',{path:'ReplicatedStorage.DemoStage',property:'Value'});
  console.log(JSON.stringify(stage));
  const report=await callTool('instance_get_property',{path:'ReplicatedStorage.DemoReport',property:'Value'});
  if(report.value) writeFileSync('roblox/out/y2_ui_samples.json',report.value);
} finally {await close();}
