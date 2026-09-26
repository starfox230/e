"""Assemble the finished chapters into the full-length film.

Produces, in out/:
  umbrella_full.mp4      every chapter concatenated (stream copy, no re-encode)
  umbrella_full.srt      subtitles shifted onto the full-film clock
  chapters.txt           YouTube chapter markers (0:00 Chapter 1 ... )
  ffmetadata.txt         embedded chapter marks for players that read them
  manifest.json          per-chapter durations, offsets and QA numbers

Usage: python3 pipeline/assemble.py [--parts N]   (--parts splits into N upload-sized files)
"""
import os
import sys
import json
import glob
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'out')


def hhmmss(t, comma=False):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    if comma:
        return f'{h:02d}:{m:02d}:{int(s):02d},{int((s - int(s)) * 1000):03d}'
    return f'{h:d}:{m:02d}:{int(s):02d}' if h else f'{m:d}:{int(s):02d}'


def chapters_present():
    out = []
    for p in sorted(glob.glob(os.path.join(OUT, 'video', 'ch*.mp4'))):
        n = int(os.path.basename(p)[2:4])
        tl = json.load(open(os.path.join(ROOT, 'work', f'ch{n:02d}', 'timeline.json')))
        out.append({'n': n, 'path': p, 'title': tl['title'], 'movement': tl['movement'],
                    'duration': tl['duration']})
    return out


def real_duration(path):
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                        '-of', 'csv=p=0', path], capture_output=True, text=True)
    return float(r.stdout.strip())


def shift_srt(src, offset, start_index):
    """Return (lines, next_index) with every cue moved by `offset` seconds."""
    lines, idx = [], start_index
    if not os.path.exists(src):
        return lines, idx
    block = []
    for raw in open(src):
        line = raw.rstrip('\n')
        if line.strip() == '' and block:
            if len(block) >= 2 and '-->' in block[1]:
                a, b = block[1].split('-->')
                def sec(t):
                    h, m, s = t.strip().replace(',', '.').split(':')
                    return int(h) * 3600 + int(m) * 60 + float(s)
                lines.append(str(idx))
                lines.append(f'{hhmmss(sec(a) + offset, True)} --> {hhmmss(sec(b) + offset, True)}')
                lines.extend(block[2:])
                lines.append('')
                idx += 1
            block = []
        else:
            block.append(line)
    return lines, idx


def main(parts=1):
    chs = chapters_present()
    if not chs:
        print('no rendered chapters')
        return 1
    os.makedirs(OUT, exist_ok=True)
    offset = 0.0
    srt_lines, idx = [], 1
    marks, meta = [], [';FFMETADATA1']
    manifest = []
    for c in chs:
        c['real'] = real_duration(c['path'])
        c['offset'] = offset
        marks.append((offset, c))
        sl, idx = shift_srt(os.path.join(OUT, 'subs', f"ch{c['n']:02d}.srt"), offset, idx)
        srt_lines += sl
        meta += ['[CHAPTER]', 'TIMEBASE=1/1000',
                 f'START={int(offset*1000)}', f'END={int((offset + c["real"])*1000)}',
                 f'title=Chapter {c["n"]}. {c["title"]}']
        manifest.append({'chapter': c['n'], 'title': c['title'], 'movement': c['movement'],
                         'offset_s': round(offset, 3), 'duration_s': round(c['real'], 3)})
        offset += c['real']

    with open(os.path.join(OUT, 'umbrella_full.srt'), 'w') as f:
        f.write('\n'.join(srt_lines))
    with open(os.path.join(OUT, 'ffmetadata.txt'), 'w') as f:
        f.write('\n'.join(meta) + '\n')
    with open(os.path.join(OUT, 'chapters.txt'), 'w') as f:
        cur_mov = None
        for t, c in marks:
            if c['movement'] and c['movement'] != cur_mov:
                cur_mov = c['movement']
                f.write(f'{hhmmss(t)} — PART {cur_mov[0]}: {cur_mov[1].upper()}\n')
            f.write(f'{hhmmss(t)} Chapter {c["n"]}. {c["title"]}\n')
    with open(os.path.join(OUT, 'manifest.json'), 'w') as f:
        json.dump({'total_s': round(offset, 2), 'total_hms': hhmmss(offset),
                   'chapters': manifest}, f, indent=1)

    groups = [chs] if parts <= 1 else [chs[i::parts] for i in range(parts)]
    if parts > 1:
        # contiguous split rather than interleaved
        per = (len(chs) + parts - 1) // parts
        groups = [chs[i * per:(i + 1) * per] for i in range(parts)]
        groups = [g for g in groups if g]
    for gi, group in enumerate(groups):
        lst = os.path.join(OUT, f'_concat{gi}.txt')
        with open(lst, 'w') as f:
            for c in group:
                f.write(f"file '{os.path.abspath(c['path'])}'\n")
        name = 'umbrella_full.mp4' if len(groups) == 1 else f'umbrella_part{gi+1}.mp4'
        dest = os.path.join(OUT, name)
        cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst]
        if len(groups) == 1:
            cmd += ['-i', os.path.join(OUT, 'ffmetadata.txt'), '-map_metadata', '1']
        cmd += ['-c', 'copy', '-movflags', '+faststart', dest]
        subprocess.run(cmd, check=True)
        os.remove(lst)
        print(f'{name}: {len(group)} chapters, {hhmmss(sum(c["real"] for c in group))}')
    print(f'total runtime {hhmmss(offset)} across {len(chs)} chapters')
    return 0


if __name__ == '__main__':
    p = int(sys.argv[sys.argv.index('--parts') + 1]) if '--parts' in sys.argv else 1
    sys.exit(main(p))
