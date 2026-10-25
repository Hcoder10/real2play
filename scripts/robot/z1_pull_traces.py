"""Copy completed trace batches off the rental, verifying every archive checksum."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--host', required=True)
    p.add_argument('--port', default='22')
    p.add_argument('--key', required=True)
    p.add_argument('--remote', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--batch', type=int, default=64)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    manifest_path = out/'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    ssh = ['ssh', '-o', 'BatchMode=yes', '-p', a.port, '-i', a.key, a.host]
    def remote(code, *args, data=None):
        command = shlex.join(['python3', '-c', code, *args])
        return json.loads(subprocess.check_output([*ssh, command], input=data, text=True))
    listing = '''import json,pathlib,sys,time
d=pathlib.Path(sys.argv[1]); done=(d/'report.json').exists()
print(json.dumps({'done':done,'files':[p.name for p in sorted((d/'success').glob('*.jsonl')) if done or time.time()-p.stat().st_mtime>60]}))'''
    archive_code = '''import hashlib,json,pathlib,sys,tarfile
d=pathlib.Path(sys.argv[1]); names=json.load(sys.stdin); dst=d/'backups'/sys.argv[2]; dst.parent.mkdir(exist_ok=True)
with tarfile.open(dst,'w:gz',compresslevel=3) as tar:
 for name in names: tar.add(d/'success'/name,arcname='success/'+name)
print(json.dumps({'path':str(dst),'sha256':hashlib.file_digest(dst.open('rb'),'sha256').hexdigest(),'bytes':dst.stat().st_size}))'''
    while True:
        state = remote(listing, a.remote)
        saved = {name for row in manifest for name in row['traces']}
        remaining = [name for name in state['files'] if name not in saved]
        if not remaining or (len(remaining) < a.batch and not state['done']):
            if state['done'] and not remaining:
                print('BACKUP_DONE', len(saved), flush=True)
                return
            time.sleep(15)
            continue
        names = remaining[:a.batch]
        name = f'traces_{len(manifest):04d}.tar.gz'
        result = remote(archive_code, a.remote, name, data=json.dumps(names))
        part = out/(name+'.part')
        subprocess.run(['scp', '-O', '-q', '-P', a.port, '-i', a.key,
                        f"{a.host}:{result['path']}", str(part)], check=True)
        with part.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != result['sha256']:
            raise ValueError(f'Archive checksum mismatch: {part}')
        part.replace(out/name)
        manifest.append({'archive':name,'traces':names,**result})
        temporary = manifest_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest, indent=2)); temporary.replace(manifest_path)
        print('SAVED', len(saved)+len(names), result['bytes'], digest, flush=True)


if __name__ == '__main__':
    main()
