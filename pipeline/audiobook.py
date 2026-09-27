"""Package the finished film as an audiobook: one AAC track per chapter.

Each track is lifted straight out of the rendered chapter video with the stream copied, not
re-encoded, so the audiobook is bit-for-bit the film's soundtrack -- narration, dialogue,
music and effects, already mastered to -15 LUFS. Taking it from the video also means the
audiobook does not need the chapter mixes, which are deleted as the render consumes them to
keep the session's disk inside its allowance. One file per chapter gives players natural
chapter navigation and keeps every track inside the file-transfer limit; a chapter whose
copied track would exceed it is the one that gets re-encoded, at --bitrate. Tracks carry
title, album, track number and the title card as cover art.

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
LIMIT = 29 * 1024 * 1024  # the file-transfer ceiling a single track has to fit under
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
    dst = os.path.join(OUT, f'{ch:02d} - {safe(title)}.m4a')
    tmp = dst + '.part.m4a'
    meta = ['-metadata', f'title=Chapter {ch}: {title}', '-metadata', f'album={ALBUM}',
            '-metadata', 'artist=Umbrella', '-metadata', 'album_artist=Umbrella',
            '-metadata', f'track={ch}/50', '-metadata', 'genre=Audiobook']
    if part:
        meta += ['-metadata', f'comment=Part {PARTS[part[0]]}: {part[1]}']
    mix = os.path.join(ROOT, 'work', f'ch{ch:02d}', 'mix.wav')
    video = os.path.join(ROOT, 'out', 'video', f'ch{ch:02d}.mp4')
    src = mix if os.path.exists(mix) else video
    # The video's track is already the mastered mix as AAC; copying it avoids a second
    # generation of lossy encoding. Only a mix, which is still WAV, has to be encoded.
    codec = ['-c:a', 'aac', '-b:a', bitrate] if src == mix else ['-c:a', 'copy']
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', src, '-i', art,
           '-map', '0:a:0', '-map', '1:v', *codec,
           '-c:v', 'mjpeg', '-disposition:v', 'attached_pic', *meta,
           '-movflags', '+faststart', tmp]
    subprocess.run(cmd, check=True)
    if os.path.getsize(tmp) > LIMIT:
        # Too big to send as one file: re-encode this chapter down instead of copying.
        cmd[cmd.index('-c:a') + 1:cmd.index('-c:a') + 2] = ['aac']
        if '-b:a' not in cmd:
            cmd[cmd.index('-c:a') + 2:cmd.index('-c:a') + 2] = ['-b:a', bitrate]
        subprocess.run(cmd, check=True)
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
