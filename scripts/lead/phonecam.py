"""Phone as the scene camera over Wi-Fi (no app, no adb): the phone's Chrome opens http://<laptop-ip>:8792/ and streams
JPEG frames to the laptop; the laptop serves the newest frame at /frame/0.jpg and /list (camserver-compatible, so
r2p.perception HttpCamera reads it unchanged). Also proxies local cameras at /frame/<i>.jpg for i >= 1 (cv2 index i-1
is NOT used: only the phone is index 0).

  python scripts/lead/phonecam.py [--port 8792]
Phone: Chrome needs a secure context for the camera. Either chrome://flags -> "Insecure origins treated as secure" ->
add http://<laptop-ip>:8792 -> Relaunch, or open the page through an https tunnel.
"""
import argparse
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!doctype html><html><head><meta name=viewport content="width=device-width,initial-scale=1">
<title>Real2Play phone camera</title><style>body{margin:0;background:#111;color:#eee;font:16px system-ui}
video{width:100%;max-height:70vh;background:#000}#s{padding:10px}button{font-size:20px;padding:12px 20px;margin:10px}</style>
</head><body><video id=v autoplay playsinline muted></video><div id=s>tap Start</div>
<button id=b>Start</button> <button id=t>Torch</button>
<script>
const v=document.getElementById('v'),s=document.getElementById('s');let track,sent=0,fails=0,t0=Date.now();const P=new URLSearchParams(location.search);const W=+(P.get('w')||960),Hh=Math.round(W*9/16),Q=+(P.get('q')||0.7);
async function start(){
  if(!navigator.mediaDevices){s.textContent='No camera API: open chrome://flags, enable "Insecure origins treated as secure" for '+location.origin+', relaunch';return;}
  const st=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'},width:{ideal:W},height:{ideal:Hh},frameRate:{ideal:30}},audio:false});
  v.srcObject=st;track=st.getVideoTracks()[0];
  try{await track.applyConstraints({advanced:[{focusMode:'manual'}]})}catch(e){}
  const c=document.createElement('canvas');
  async function loop(){
    if(v.videoWidth){c.width=W;c.height=Hh;c.getContext('2d').drawImage(v,0,0,W,Hh);
      const b=await new Promise(r=>c.toBlob(r,'image/jpeg',Q));
      const ac=new AbortController();const to=setTimeout(()=>ac.abort(),1500);
      try{await fetch('/upload',{method:'POST',body:b,signal:ac.signal,cache:'no-store'});sent++;}catch(e){fails++;}finally{clearTimeout(to);}
      s.textContent=`streaming ${v.videoWidth}x${v.videoHeight}  ${(sent*1000/(Date.now()-t0)).toFixed(1)} fps, ${fails} retries - keep this tab open, screen on`;}
    setTimeout(loop,30);}
  loop();}
document.getElementById('b').onclick=()=>start().catch(e=>s.textContent='error: '+e);
document.getElementById('t').onclick=async()=>{try{const on=!(track.getSettings().torch);await track.applyConstraints({advanced:[{torch:on}]})}catch(e){}};
</script></body></html>"""

latest = {"jpg": None, "t": 0.0, "n": 0}
lock = threading.Lock()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, PAGE.encode(), "text/html")
        if self.path == "/list":
            with lock:
                info = {"0": {"source": "phone", "age_s": round(time.time() - latest["t"], 2), "frames": latest["n"]}} \
                    if latest["jpg"] else {}
            return self._send(200, json.dumps(info).encode(), "application/json")
        if self.path.startswith("/frame/0"):
            with lock:
                jpg = latest["jpg"]
            return self._send(200, jpg, "image/jpeg") if jpg else self._send(404, b"no frame yet", "text/plain")
        self._send(404, b"", "text/plain")

    def do_POST(self):
        if self.path != "/upload":
            return self._send(404, b"", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        if not 0 < n < 8_000_000:
            return self._send(413, b"", "text/plain")
        body = self.rfile.read(n)
        with lock:
            latest.update(jpg=body, t=time.time(), n=latest["n"] + 1)
        self._send(204, b"", "text/plain")


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8792)
    a = ap.parse_args()
    print(f"phonecam: open http://{lan_ip()}:{a.port}/ on the phone (frames at http://127.0.0.1:{a.port}/frame/0.jpg)",
          flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
