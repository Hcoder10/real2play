"""Extract the observed 1080p simulator viewport from Y2's Studio captures."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ffmpeg',type=Path,required=True)
    args=parser.parse_args()
    probe=args.ffmpeg.with_name('ffprobe'+args.ffmpeg.suffix)
    results=[]
    for name in ('lobby','cap','bowl'):
        path=ROOT/f'out/demo/roblox_{name}.mp4'
        original=path.with_stem(path.stem+'_studio')
        if not original.exists(): path.replace(original)
        data=json.loads(subprocess.check_output([str(probe),'-v','error','-show_streams','-show_format','-of','json',str(original)],text=True))
        stream=data['streams'][0]
        assert (stream['width'],stream['height'])==(1920,1120), 'Capture layout changed; inspect before cropping'
        subprocess.run([str(args.ffmpeg),'-hide_banner','-loglevel','error','-i',str(original),
            '-vf','crop=1426:802:28:226,scale=1920:1080','-c:v','libx264','-threads','2',
            '-preset','veryfast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart','-an','-y',str(path)],check=True)
        duration=float(data['format']['duration'])
        assert 15<=duration<=25.1
        results.append({'file':str(path.relative_to(ROOT)),'duration_s':duration,'width':1920,'height':1080,
            'codec':'h264','source':'actual Roblox Studio Play, gdigrab; viewport crop; real-time',
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        subprocess.run([str(args.ffmpeg),'-hide_banner','-loglevel','error','-ss',str(duration/2),'-i',str(path),
            '-frames:v','1','-y',str(path.with_suffix('.png'))],check=True)
    (ROOT/'roblox/out/y2_videos.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))

if __name__=='__main__': main()
