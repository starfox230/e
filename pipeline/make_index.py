"""Rewrite story/storyboard.md as an index of what was actually written.

The storyboard was a plan; the chapters diverged from it as they were written, and a plan
that no longer matches the film is worse than no plan. This reads the titles, Part openers,
dates and opening narration out of the scripts.
"""
import os
import re
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = ['# The film, as written', '',
       'Generated from `script/chNN.txt` by `pipeline/make_index.py`. Fifty chapters, six Parts.', '']
for path in sorted(glob.glob(os.path.join(ROOT, 'script', 'ch*.txt'))):
    lines = open(path).read().split('\n')
    num = title = movement = first_loc = opening = None
    words = 0
    for l in lines:
        s = l.strip()
        m = re.match(r'^#\s*(\d+)\s*\|\s*(.+)$', s)
        if m:
            num, title = int(m.group(1)), m.group(2)
        elif s.startswith('@movement'):
            movement = s[len('@movement'):].strip()
        elif s.startswith('[loc]') and first_loc is None:
            first_loc = s[len('[loc]'):].strip()
        elif s.startswith('N:') and opening is None and len(s) > 80:
            opening = s[2:].strip()
        if re.match(r'^[A-Z]+:', s):
            words += len(s.split()) - 1
    if movement:
        out += ['', f'## Part {movement}', '']
    out.append(f'**{num}. {title}** — {first_loc or ""}  ')
    out.append(f'{(opening or "")[:240]}…  ')
    out.append(f'*{words:,} spoken words*')
    out.append('')
total = 0
for path in sorted(glob.glob(os.path.join(ROOT, 'script', 'ch*.txt'))):
    for l in open(path):
        if re.match(r'^[A-Z]+:', l.strip()):
            total += len(l.split()) - 1
out += ['---', '', f'**Total: {total:,} spoken words** (about {total / 145 / 60:.1f} hours of narration).']
open(os.path.join(ROOT, 'story', 'storyboard.md'), 'w').write('\n'.join(out))
print(f'{total:,} spoken words indexed')
