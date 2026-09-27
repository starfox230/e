"""Re-encode each rendered chapter small enough to send through the conversation.

The 1080p masters are 70-95 MB a chapter and the file channel stops at 30 MiB, so a chapter
that has to travel this way gets its own encode: 960x540, and a video bitrate solved from the
chapter's own length so the result lands just under the cap. The picture is slow pans over
still art, which is the easiest thing in the world for x264 to carry at a low rate, but these
are a delivery copy and not the master -- the 1080p files stay in out/video.

Usage: python3 pipeline/make_small.py [--cap-mib 28] [--jobs 3] [--height 540] [N ...]
"""
import os
import sys
import glob
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'out', 'small')
AUDIO_K = 96


def duration(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                          '-of', 'default=nw=1:nk=1', path], capture_output=True, text=True)
    return float(out.stdout.strip())


def encode(job):
    src, cap, height = job
    ch = os.path.basename(src)[2:4]
    dst = os.path.join(OUT, f'ch{ch}.mp4')
    secs = duration(src)
    # Solve the video rate from the cap, leaving the audio and about 2% of container overhead.
    kbps = int((cap * 8192 / secs) * 0.98) - AUDIO_K
    for attempt in range(3):
        subprocess.run(
            ['ffmpeg', '-y', '-loglevel', 'error', '-i', src,
             '-vf', f'scale=-2:{height}', '-c:v', 'libx264', '-preset', 'veryfast',
             '-b:v', f'{kbps}k', '-maxrate', f'{int(kbps * 1.4)}k', '-bufsize', f'{kbps * 4}k',
             '-tune', 'stillimage', '-pix_fmt', 'yuv420p',
             '-c:a', 'aac', '-b:a', f'{AUDIO_K}k', '-movflags', '+faststart', dst], check=True)
        if os.path.getsize(dst) <= cap * 1024 * 1024:
            break
        kbps = int(kbps * 0.9)                    # overshot: the cap is what has to hold
    return dst, os.path.getsize(dst) / 1024 / 1024, kbps


def main():
    argv = sys.argv[1:]
    cap, jobs_n, height = 28, 3, 540
    for flag, name in (('--cap-mib', 'cap'), ('--jobs', 'jobs_n'), ('--height', 'height')):
        if flag in argv:
            i = argv.index(flag)
            value = int(argv[i + 1])
            cap, jobs_n, height = ((value, jobs_n, height) if name == 'cap' else
                                   (cap, value, height) if name == 'jobs_n' else
                                   (cap, jobs_n, value))
            del argv[i:i + 2]        # a flag's value is not a chapter number
    only = {int(a) for a in argv if a.isdigit()}
    os.makedirs(OUT, exist_ok=True)
    srcs = sorted(glob.glob(os.path.join(ROOT, 'out', 'video', 'ch*.mp4')))
    if only:
        srcs = [s for s in srcs if int(os.path.basename(s)[2:4]) in only]
    if not srcs:
        print('nothing rendered yet')
        return 1
    with ThreadPoolExecutor(max_workers=jobs_n) as ex:
        for dst, mb, kbps in ex.map(encode, [(s, cap, height) for s in srcs]):
            print(f'  {os.path.basename(dst)}  {mb:.1f} MB  ({kbps} kbps video)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
