"""Delete and re-render images whose scene type changed, so art stays consistent with the renderer."""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse import parse
from audio import shot_id

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = ('basement', 'freezer', 'hvac', 'cryogenic', 'cryo', 'cradle', 'frosted', 'capsule',
        'loading dock', 'server', 'data centre', 'data center', 'plant room', 'concrete')

removed = 0
for f in sorted(glob.glob(os.path.join(ROOT, 'script', 'ch*.txt'))):
    ch = parse(f)
    for e in (e for e in ch.events if e.kind == 'img'):
        low = e.text.lower()
        if any(k in low for k in KEYS):
            p = os.path.join(ROOT, 'work', 'images', f'{shot_id(ch.number, e.text)}.png')
            if os.path.exists(p):
                os.remove(p)
                removed += 1
print(f'{removed} stale images removed; re-run build_images.py')
