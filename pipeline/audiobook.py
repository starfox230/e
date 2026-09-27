"""Package the finished chapter mixes as an audiobook: one AAC track per chapter.

The mixes are the film's soundtrack as-is -- narration, dialogue, music and effects, already
mastered to -15 LUFS -- so the audiobook is exactly what the film sounds like. One file per
chapter keeps every track under the 30 MiB file-transfer limit (the longest chapter comes to
about 18 MB at 128 kbps) and gives players natural chapter navigation. Tracks carry title,
album, track number and the title card as cover art.

Usage: python3 pipeline/audiobook.py [--bitrate 128k] [--jobs 3]
"""
import os
import re
import sys
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'out', 'audiobook')
ALBUM = 'UMBRELLA: What If Hitler Was Reborn as Albert Wesker'
PARTS = {'I': 'One', 'II': 'Two', 'III': 'Three', 'IV': 'Four', 'V': 'Five', 'VI': 'Six'}


def cover(path):
    """The series title card, centred on a square canvas, since players expect square art."""
    import graphics
    from PIL import Image
    card = graphics.main_title()
    side = 1400
    bg = card.getpixel((8, 8))
    sq = Image.new('RGB', (side, side), bg)
    card = card.resize((side, int(side * card.height / card.width)))
    sq.paste(card, (0, (side - card.height) // 2))
    sq.save(path, quality=92)


def safe(name):
    return re.sub(r'[\\/:*?"<>|]+', '', name).strip()


def encode(job):
    ch, title, part, art, bitrate = job
    src = os.path.join(ROOT, 'work', f'ch{ch:02d}', 'mix.wav')
    dst = os.path.join(OUT, f'{ch:02d} - {safe(title)}.m4a')
    tmp = dst + '.part.m4a'
    meta = ['-metadata', f'title=Chapter {ch}: {title}', '-metadata', f'album={ALBUM}',
            '-metadata', 'artist=Umbrella', '-metadata', 'album_artist=Umbrella',
            '-metadata', f'track={ch}/50', '-metadata', 'genre=Audiobook']
    if part:
        meta += ['-metadata', f'comment=Part {PARTS[part[0]]}: {part[1]}']
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', src, '-i', art,
                    '-map', '0:a', '-map', '1:v', '-c:a', 'aac', '-b:a', bitrate,
                    '-c:v', 'mjpeg', '-disposition:v', 'attached_pic', *meta,
                    '-movflags', '+faststart', tmp], check=True)
    os.replace(tmp, dst)
    return dst


def main():
    bitrate = sys.argv[sys.argv.index('--bitrate') + 1] if '--bitrate' in sys.argv else '128k'
    jobs_n = int(sys.argv[sys.argv.index('--jobs') + 1]) if '--jobs' in sys.argv else 3
    os.makedirs(OUT, exist_ok=True)
    art = os.path.join(OUT, 'cover.jpg')
    cover(art)
    jobs = []
    for ch in range(1, 51):
        tl = json.load(open(os.path.join(ROOT, 'work', f'ch{ch:02d}', 'timeline.json')))
        jobs.append((ch, tl['title'], tl['movement'], art, bitrate))
    with ThreadPoolExecutor(max_workers=jobs_n) as ex:
        for dst in ex.map(encode, jobs):
            print(f'  {os.path.basename(dst)}  {os.path.getsize(dst) / 1e6:.1f} MB', flush=True)


if __name__ == '__main__':
    main()
