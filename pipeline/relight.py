"""Re-render the scene art that came out too dark to read on screen.

art.grade() gained an exposure floor after the first art pass had already been packed to
JPEG, so the frames rendered before it are still flat black. This finds them by measuring
the finished files, re-renders just those through the current grade, and writes them back
at the same JPEG quality pack_images.py uses. Everything else is left alone, so the pass is
cheap and the art direction elsewhere is untouched.

Usage: python3 pipeline/relight.py [--dry-run] [--workers N]
"""
import os
import sys
import glob
import json
import hashlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES = os.path.join(ROOT, 'work', 'images')
QUALITY = 92


def prompts():
    """Every shot id in the film, with the prompt it was rendered from."""
    out = {}
    for tl_path in sorted(glob.glob(os.path.join(ROOT, 'work', 'ch*', 'timeline.json'))):
        for shot in json.load(open(tl_path))['shots']:
            out[shot['id']] = shot['prompt']
    return out


def too_dark(path, floor):
    a = np.asarray(Image.open(path).convert('L'), dtype=np.float32)
    return float(np.percentile(a, 99)) < floor * 255


def relight(job):
    sid, prompt, path = job
    import art
    if sid.endswith('_v1'):
        # the second composition of a shot is seeded from the id plus a suffix, the same
        # way the first art pass built it, so the picture stays the one that was cut in
        base = sid[:-3]
        seed = int(hashlib.sha1((base + 'v1').encode()).hexdigest()[:8], 16)
        img = art.Scene(art.strip_tags(prompt), seed).compose()
    else:
        img = art.render(prompt, sid)
    img.convert('RGB').save(path, quality=QUALITY, subsampling=1, optimize=True)
    a = np.asarray(img.convert('L'), dtype=np.float32)
    return sid, float(np.percentile(a, 99))


def main():
    import art
    floor = art.FLOOR_P99
    look = prompts()
    jobs = []
    for path in sorted(glob.glob(os.path.join(IMAGES, '*.jpg'))):
        sid = os.path.basename(path)[:-4]
        base = sid[:-3] if sid.endswith('_v1') else sid
        if base not in look:
            continue
        if too_dark(path, floor):
            jobs.append((sid, look[base], path))
    print(f'{len(jobs)} images below the p99 floor of {floor:.2f} ({floor*255:.0f}/255)')
    if '--dry-run' in sys.argv:
        for sid, _, _ in jobs:
            print('  ', sid)
        return 0
    workers = 4
    if '--workers' in sys.argv:
        workers = int(sys.argv[sys.argv.index('--workers') + 1])
    with ProcessPoolExecutor(max_workers=workers) as ex:
        done = list(ex.map(relight, jobs, chunksize=1))
    still = [(s, p) for s, p in done if p < floor * 255]
    for sid, p in sorted(done, key=lambda x: x[1])[:10]:
        print(f'  {sid:34s} p99 now {p:6.1f}')
    print(f'relit {len(done)} images; {len(still)} still under the floor')
    return 0


if __name__ == '__main__':
    sys.exit(main())
