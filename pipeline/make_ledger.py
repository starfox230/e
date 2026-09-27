"""Rebuild story/ledger.md from the scripts, so canon cannot drift from what was written.

The old ledger was written first and the chapters were written against it; after the rebuild
the chapters are the source of truth. This reads the [system] panels and [loc] lines out of
script/chNN.txt and prints what each chapter actually claims: its dates, its Points, its
level, and every roster or tier line.
"""
import os
import re
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEEP = re.compile(r'^(Points|Balance|Level|Tier|Roster|Absolute|Integrated|Personnel|Population|'
                  r'Market capitalisation|Viral Affinity|Presence|Intelligence|Strength|Released|'
                  r'Territories?|Facilities|Sentiment|Share|Enumerated|Human dead|Casualt)', re.I)

out = ['# Continuity ledger', '',
       'Generated from the scripts by `pipeline/make_ledger.py`. The chapters are canon; this',
       'is a reading of them. Re-run it after any change to a `[system]` panel.', '']
for path in sorted(glob.glob(os.path.join(ROOT, 'script', 'ch*.txt'))):
    lines = open(path).read().split('\n')
    num = title = None
    dates, facts, panels = [], [], []
    inside = None
    for l in lines:
        s = l.strip()
        m = re.match(r'^#\s*(\d+)\s*\|\s*(.+)$', s)
        if m:
            num, title = int(m.group(1)), m.group(2)
        elif s.startswith('[loc]'):
            d = re.search(r'—\s*(.+)$', s)
            if d:
                dates.append(d.group(1).strip())
        elif s.startswith('[system]'):
            inside = s[len('[system]'):].strip()
            panels.append(inside)
        elif s == '[/system]':
            inside = None
        elif inside is not None and KEEP.match(s):
            facts.append(s)
    out.append(f'## {num}. {title}')
    if dates:
        out.append(f'- **When:** {dates[0]}' + (f' — {dates[-1]}' if len(dates) > 1 else ''))
    if panels:
        out.append('- **Panels:** ' + ' · '.join(dict.fromkeys(panels)))
    for f in facts:
        out.append(f'  - {f}')
    out.append('')
open(os.path.join(ROOT, 'story', 'ledger.md'), 'w').write('\n'.join(out))
print(f'{len(out)} lines written')
