"""Verify every [sfx]/[amb]/[music] cue in the scripts has a rendered asset. Exit 1 if not."""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse import parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
missing = {}
for f in sorted(glob.glob(os.path.join(ROOT, 'script', 'ch*.txt'))):
    ch = parse(f)
    for e in ch.events:
        if e.kind in ('sfx', 'amb', 'amb+'):
            if e.name == 'stop':
                continue
            p = os.path.join(ROOT, 'assets', 'sfx', f'{e.name}.wav')
        elif e.kind == 'music':
            if e.name == 'stop':
                continue
            p = os.path.join(ROOT, 'assets', 'music', f'{e.name}.wav')
        else:
            continue
        if not os.path.exists(p):
            missing.setdefault(os.path.basename(f), set()).add(e.name)
for f, names in sorted(missing.items()):
    print(f, sorted(names))
print('MISSING' if missing else 'all cues present')
sys.exit(1 if missing else 0)
