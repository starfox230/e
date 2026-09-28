"""Re-encode each rendered chapter small enough to send through the conversation.

The 1080p masters are 70-95 MB a chapter and the file channel stops at 30 MiB, so a chapter
that has to travel this way gets its own encode at 720p. The video bitrate is solved from the
chapter's own length against the cap, which means a long chapter would otherwise be starved:
eighteen minutes inside 28 MiB leaves about 110 kbps, which is thin spread over 1280x720. So a
chapter whose solved rate falls under FLOOR is split into as many parts as it takes to clear
it, each part encoded separately and named chNN_pM. Splitting buys real bitrate where dropping
the resolution would have bought it by giving up detail, and the picture here is slow pans over
still art, which keeps detail and loses little to motion.

These are delivery copies, not masters; the 1080p files stay in out/video.

Usage: python3 pipeline/make_small.py [--cap-mib 28] [--jobs 3] [--height 720] [N ...]
"""
import os
import sys
import glob
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'out', 'small')
AUDIO_K = 96
FLOOR = 170      # kbps of video below which 720p starts to visibly soften, so split instead


def duration(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                          '-of', 'default=nw=1:nk=1', path], capture_output=True, text=True)
    return float(out.stdout.strip())


def rate_for(secs, cap):
    """Video kbps that fills the cap for a clip this long, after audio and container overhead."""
    return int((cap * 8192 / secs) * 0.98) - AUDIO_K


def encode_span(src, dst, start, secs, kbps, height, cap):
    """One output file: either a whole chapter or one part of a split one."""
    for _ in range(3):
        cmd = ['ffmpeg', '-y', '-loglevel', 'error']
        if start:
            cmd += ['-ss', f'{start:.3f}']
        cmd += ['-i', src]
        if secs:
            cmd += ['-t', f'{secs:.3f}']
        cmd += ['-vf', f'scale=-2:{height}', '-c:v', 'libx264', '-preset', 'veryfast',
                '-b:v', f'{kbps}k', '-maxrate', f'{int(kbps * 1.4)}k', '-bufsize', f'{kbps * 4}k',
                '-tune', 'stillimage', '-pix_fmt', 'yuv420p',
                '-c:a', 'aac', '-b:a', f'{AUDIO_K}k', '-movflags', '+faststart', dst]
        subprocess.run(cmd, check=True)
        if os.path.getsize(dst) <= cap * 1024 * 1024:
            return kbps
        kbps = int(kbps * 0.9)                    # overshot: the cap is what has to hold
    return kbps


def encode(job):
    src, cap, height = job
    ch = os.path.basename(src)[2:4]
    total = duration(src)
    parts = 1
    while rate_for(total / parts, cap) < FLOOR and parts < 4:
        parts += 1
    made = []
    if parts == 1:
        dst = os.path.join(OUT, f'ch{ch}.mp4')
        kbps = encode_span(src, dst, 0, None, rate_for(total, cap), height, cap)
        made.append((dst, os.path.getsize(dst) / 1024 / 1024, kbps))
    else:
        span = total / parts
        kbps = rate_for(span, cap)
        for i in range(parts):
            dst = os.path.join(OUT, f'ch{ch}_p{i + 1}.mp4')
            secs = None if i == parts - 1 else span
            got = encode_span(src, dst, i * span, secs, kbps, height, cap)
            made.append((dst, os.path.getsize(dst) / 1024 / 1024, got))
    return made


def main():
    argv = sys.argv[1:]
    cap, jobs_n, height = 28, 3, 720
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
        for made in ex.map(encode, [(s, cap, height) for s in srcs]):
            for dst, mb, kbps in made:
                print(f'  {os.path.basename(dst)}  {mb:.1f} MB  ({kbps} kbps video, {height}p)',
                      flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
