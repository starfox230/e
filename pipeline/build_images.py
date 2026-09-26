"""Render every chapter's scene images in parallel across all cores."""
import os
import sys
import glob
import hashlib
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse import parse          # noqa: E402
from audio import shot_id        # noqa: E402
import art                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'work', 'images')


VARIANTS = int(os.environ.get('ART_VARIANTS', '2'))   # base + N-1 alternate compositions


def jobs(force=False, variants=None):
    """One job per shot per variant. Variants are alternate compositions of the same
    prompt from a different seed, so a long shot can cut between them instead of holding
    one picture for a minute."""
    variants = VARIANTS if variants is None else variants
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, 'script', 'ch*.txt'))):
        ch = parse(f)
        for e in (e for e in ch.events if e.kind == 'img'):
            sid = shot_id(ch.number, e.text)
            for v in range(variants):
                name = sid if v == 0 else f'{sid}_v{v}'
                p = os.path.join(OUT, f'{name}.png')
                if force or not (os.path.exists(p) or os.path.exists(p[:-4] + '.jpg')):
                    out.append((e.text, sid, p, v))
    return out


def render_one(job):
    prompt, sid, path, variant = job
    key = sid if variant == 0 else f'{sid}|v{variant}'
    seed = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)
    art.Scene(art.strip_tags(prompt), seed).compose().save(path, optimize=True)
    return path


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    js = jobs('--force' in sys.argv)
    print(f'{len(js)} images to render', flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=4) as ex:
        for _ in ex.map(render_one, js, chunksize=4):
            done += 1
            if done % 25 == 0:
                print(f'  {done}/{len(js)}', flush=True)
    print('images done', flush=True)
