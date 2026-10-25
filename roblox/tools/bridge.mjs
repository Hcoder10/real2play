// Minimal MCP-over-HTTP client for the existing Studio bridge. No npm dependency.
const base = process.env.STUDIO_BRIDGE_URL || 'http://localhost:3000';
let session, nextId = 1, connected = false;
async function rpc(method, params, notification = false) {
  const headers = { 'Content-Type': 'application/json', Accept: 'application/json, text/event-stream' };
  if (session) headers['mcp-session-id'] = session;
  const response = await fetch(`${base}/mcp`, { method: 'POST', headers,
    body: JSON.stringify({ jsonrpc: '2.0', ...(notification ? {} : { id: nextId++ }), method, params }),
    signal: AbortSignal.timeout(125000) });
  session ||= response.headers.get('mcp-session-id');
  if (!response.ok) throw new Error(`bridge HTTP ${response.status}`);
  const body = await response.text();
  if (notification || !body) return;
  let value;
  if (body.trim().startsWith('{')) value = JSON.parse(body);
  else {
    const lines = body.split('\n').filter(line => line.startsWith('data:'));
    value = lines.map(line => JSON.parse(line.slice(5).trim())).find(item => item.result || item.error);
  }
  if (!value || value.error) throw new Error(JSON.stringify(value?.error || body));
  return value.result;
}
export async function connect() {
  if (connected) return;
  await rpc('initialize', { protocolVersion: '2025-03-26', capabilities: {}, clientInfo: { name: 'real2play', version: '1.0' } });
  await rpc('notifications/initialized', {}, true);
  const health = await (await fetch(`${base}/health`)).json();
  const candidate = (health.session_list || []).find(item => process.env.STUDIO_BRIDGE_TOKEN
    ? item.token === process.env.STUDIO_BRIDGE_TOKEN
    : process.env.BRIDGE === 'server' ? item.token.endsWith('-server') : !item.token.endsWith('-server'));
  if (!candidate) throw new Error('Studio bridge has no requested session');
  const out = await rpc('tools/call', { name: 'session_connect', arguments: { token: candidate.token } });
  const result = JSON.parse(out.content.filter(c => c.type === 'text').map(c => c.text).join('\n'));
  if (!result.success) throw new Error('Studio session connection failed');
  connected = true;
}
export function quote(source) {
  let equals = '';
  while ((source + ']').includes(`]${equals}]`)) equals += '=';
  return `[${equals}[${source}]${equals}]`;
}
export async function run(code, timeout = 30000) {
  await connect();
  const out = await rpc('tools/call', { name: 'execution_run_luau', arguments: { code, timeout_ms: timeout } });
  const result = JSON.parse(out.content.filter(c => c.type === 'text').map(c => c.text).join('\n'));
  if (!result.success) throw new Error(JSON.stringify(result.error));
  return result.data && 'value' in result.data ? result.data.value : result.data;
}
export async function callTool(name, args = {}) {
  await connect();
  const out = await rpc('tools/call', { name, arguments: args });
  const result = JSON.parse(out.content.filter(c => c.type === 'text').map(c => c.text).join('\n'));
  if (!result.success) throw new Error(JSON.stringify(result.error));
  return result.data;
}
export async function close() {
  if (session) await fetch(`${base}/mcp`, { method: 'DELETE', headers: { 'mcp-session-id': session } }).catch(() => {});
  session = undefined; connected = false;
}
