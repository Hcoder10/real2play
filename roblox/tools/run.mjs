import { readFileSync } from 'node:fs';
import { run, quote, close } from './bridge.mjs';
const [file, timeout = '30000', argsFile] = process.argv.slice(2);
try {
  const prefix = argsFile ? `_G.R2P_ARGS=game:GetService("HttpService"):JSONDecode(${quote(readFileSync(argsFile,'utf8'))})\n` : '';
  console.log(JSON.stringify(await run(prefix + readFileSync(file, 'utf8'), Number(timeout)), null, 2));
} catch (error) { console.error(error.message); process.exitCode=1; }
finally { await close(); }
