"""Render a chapter's video by splitting it into time segments across all cores.

Each worker composes its own segment to a raw-video-only MP4; the segments are then
concatenated without re-encoding and muxed with the chapter's mixed audio, so the
picture stays on exactly the same clock as the sound.

Usage: python3 pipeline/render_chapter.py 1 [2 3 ...]      (no args = all mixed chapters)
       --workers N      (default 4)
       --force          re-render even if out/video/chNN.mp4 exists
"""
import os
import sys
import json
import math
import glob
import subprocess
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPS = 30
PRESET = os.environ.get('X264_PRESET', 'faster')
CRF = os.environ.get('X264_CRF', '23')


def render_segment(args):
    """Compose frames [f0, f1) of a chapter into a video-only MP4."""
    chapter, f0, f1, path = args
    import video
    comp = video.Composer(chapter)
    n = f1 - f0
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
           '-s', f'{video.W}x{video.H}', '-r', str(FPS), '-i', '-',
           '-an', '-c:v', 'libx264', '-preset', PRESET, '-crf', CRF, '-tune', 'film',
           '-pix_fmt', 'yuv420p', '-g', str(FPS * 4), '-x264-params', 'scenecut=0:open-gop=0', path]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for k in range(n):
        t = (f0 + k) / FPS
        ff.stdin.write(comp.frame(t).tobytes())
    ff.stdin.close()
    ff.wait()
    if ff.returncode:
        raise RuntimeError(f'segment {path} failed')
    return path


def render(chapter, workers=4, force=False):
    work = os.path.join(ROOT, 'work', f'ch{chapter:02d}')
    tl = json.load(open(os.path.join(work, 'timeline.json')))
    out_dir = os.path.join(ROOT, 'out', 'video')
    os.makedirs(out_dir, exist_ok=True)
    final = os.path.join(out_dir, f'ch{chapter:02d}.mp4')
    if os.path.exists(final) and not force:
        return final
    total = int(math.ceil(tl['duration'] * FPS))
    seg_dir = os.path.join(work, 'segments')
    os.makedirs(seg_dir, exist_ok=True)
    per = int(math.ceil(total / workers))
    jobs = []
    for i in range(workers):
        f0, f1 = i * per, min(total, (i + 1) * per)
        if f0 >= f1:
            continue
        jobs.append((chapter, f0, f1, os.path.join(seg_dir, f'seg{i:02d}.mp4')))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(render_segment, jobs))
    lst = os.path.join(seg_dir, 'list.txt')
    with open(lst, 'w') as f:
        for _, _, _, p in jobs:
            f.write(f"file '{os.path.abspath(p)}'\n")
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst,
                    '-i', os.path.join(work, 'mix.wav'),
                    '-map', '0:v', '-map', '1:a', '-c:v', 'copy',
                    '-c:a', 'aac', '-b:a', '192k', '-shortest',
                    '-movflags', '+faststart', final], check=True)
    for _, _, _, p in jobs:
        os.remove(p)
    os.remove(lst)
    return final


if __name__ == '__main__':
    args = [int(a) for a in sys.argv[1:] if a.isdigit()]
    workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 4
    force = '--force' in sys.argv
    if not args:
        args = sorted(int(os.path.basename(os.path.dirname(p))[2:])
                      for p in glob.glob(os.path.join(ROOT, 'work', 'ch*', 'mix.wav')))
    for ch in args:
        import time
        t0 = time.time()
        p = render(ch, workers, force)
        dur = json.load(open(os.path.join(ROOT, 'work', f'ch{ch:02d}', 'timeline.json')))['duration']
        print(f'ch{ch:02d} {p} {dur/60:.1f} min in {time.time()-t0:.0f}s '
              f'({dur/max(1e-6, time.time()-t0):.2f}x realtime)', flush=True)
