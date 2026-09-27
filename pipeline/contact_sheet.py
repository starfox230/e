"""Contact sheet of the newest generated frames, for reviewing the art as it arrives.

Each run shows only frames finished since the previous sheet (tracked in .last_sheet in the
image directory), labelled with chapter and shot so a bad frame can be traced to its prompt.

Usage: python3 pipeline/contact_sheet.py OUT.jpg [image-dir] [--max N] [--all]
                 [--story] [--cols N] [--chapters A-B]

--story lays the frames out in story order instead of the order they finished, and --chapters
limits the sheet to a range, so a part of the film can be reviewed as it will be watched.
"""
import os
import sys
import glob
import json

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    argv = sys.argv[1:]
    for flag in ('--max', '--cols', '--chapters'):
        if flag in argv:
            i = argv.index(flag)
            del argv[i:i + 2]        # the value is not a path
    args = [a for a in argv if not a.startswith('--')]
    out = args[0]
    d = args[1] if len(args) > 1 else os.path.join(ROOT, 'work', 'images_flux')
    most = int(sys.argv[sys.argv.index('--max') + 1]) if '--max' in sys.argv else 16
    cols = int(sys.argv[sys.argv.index('--cols') + 1]) if '--cols' in sys.argv else 4
    span = None
    if '--chapters' in sys.argv:
        lo, _, hi = sys.argv[sys.argv.index('--chapters') + 1].partition('-')
        span = (int(lo), int(hi or lo))
    mark = os.path.join(d, '.last_sheet')
    since = 0.0 if '--all' in sys.argv or not os.path.exists(mark) else float(open(mark).read() or 0)
    frames = [f for f in glob.glob(os.path.join(d, '*.jpg'))
              if not os.path.basename(f).startswith('.') and os.path.getmtime(f) > since]
    if span:
        frames = [f for f in frames if span[0] <= int(os.path.basename(f)[1:3]) <= span[1]]
    frames.sort(key=os.path.basename if '--story' in sys.argv else os.path.getmtime)
    if not frames:
        print('no new frames')
        return 1
    newest = max(os.path.getmtime(f) for f in frames)
    if len(frames) > most:
        # spread the picks across everything new rather than showing only the last few
        step = len(frames) / most
        frames = [frames[int(i * step)] for i in range(most)]
    titles = {}
    for p in glob.glob(os.path.join(ROOT, 'work', 'ch*', 'timeline.json')):
        tl = json.load(open(p))
        titles[tl["chapter"]] = tl["title"]
    W, H, cap = 480, 270, 22
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new('RGB', (W * cols, (H + cap) * rows), (12, 12, 14))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    except Exception:
        font = ImageFont.load_default()
    for k, f in enumerate(frames):
        x, y = (k % cols) * W, (k // cols) * (H + cap)
        sheet.paste(Image.open(f).convert('RGB').resize((W, H)), (x, y))
        name = os.path.basename(f)[:-4]
        ch = int(name[1:3])
        label = f'Ch {ch} · {titles.get(ch, "")}'[:52] + ('  (alt)' if name.endswith('_v1') else '')
        draw.text((x + 6, y + H + 3), label, fill=(210, 210, 214), font=font)
    sheet.save(out, quality=88)
    if not (span or '--all' in sys.argv):
        open(mark, 'w').write(str(newest))
    print(f'{len(frames)} frames on the sheet')
    return 0


if __name__ == '__main__':
    sys.exit(main())
