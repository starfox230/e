"""Generate the film's scene art in anime style, straight from the scripts.

The brief asked for anime rather than photoreal, and on a side-by-side of six checkpoints the
Horde offers, AAM XL drew both halves of this film best: the protagonist the way the story
needs him -- slicked-back blond hair, black wraparound glasses, long black coat, flat cel
shading, hard key light -- and, unusually for an anime checkpoint, architecture and weather
that can carry an establishing shot. Animagine XL draws an equally good figure and a poor
building, so it stands as the fallback.

Two things about these models shape this file:

  * It is tag-trained. "scenery, no humans" makes it draw a background plate; without those
    tags it puts a person in the middle of every establishing shot it is given. So prompts are
    sorted into character shots and background shots and given different tails.
  * It is tuned on adult art like most anime checkpoints, and the Horde's filter throws away
    anything it does not like, so the negative prompt carries the usual guards and the request
    keeps the worker-side censor on. A frame that comes back empty is retried with a new seed.

Shot ids match audio.shot_id (sha1 of the prompt text), so a frame generated now lines up with
whatever timeline is built later -- which means art can be generated while chapters are still
being written.

Usage: python3 pipeline/gen_anime.py [--upto N] [--inflight N] [--reverse] [--limit N] [--report]
"""
import os
import re
import sys
import glob
import json
import time
import signal
import hashlib
import threading
import queue
import urllib.error
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import horde
import gen_images

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get('GEN_OUT') or os.path.join(ROOT, 'work', 'images_anime')
W, H = 1344, 768
KEY = os.environ.get('HORDE_KEY') or horde.ANON

# The Horde's anonymous queue for any one checkpoint drains at about two frames a minute,
# so a second process on a different checkpoint roughly doubles the rate. ANIME_MODEL picks
# which one this process asks for; the other is its fallback.
CHECKPOINTS = {
    'aam': {'models': ['AAM XL'], 'steps': 28, 'cfg_scale': 5.5, 'sampler_name': 'k_euler_a'},
    'animagine': {'models': ['Animagine XL'], 'steps': 28, 'cfg_scale': 6.0, 'sampler_name': 'k_euler_a'},
    'novaflat': {'models': ['Nova Flat XL'], 'steps': 28, 'cfg_scale': 5.5, 'sampler_name': 'k_euler_a'},
}
_primary = os.environ.get('ANIME_MODEL', 'aam')
MODEL = CHECKPOINTS[_primary]
FALLBACK = CHECKPOINTS['animagine' if _primary != 'animagine' else 'aam']
horde.MODELS['anime'] = MODEL
horde.MODELS['anime2'] = FALLBACK

CHAR_TAIL = ('modern anime film still, cel shaded, clean lineart, dramatic cinematic lighting, '
             'detailed background, rich colour, anime key visual, absurdres')
BG_TAIL = ('scenery, no humans, anime background art, absurdres, highly detailed background, '
           'cinematic composition, dramatic lighting, rich colour')
NEG = ('text, watermark, signature, logo, letters, deformed, extra fingers, extra limbs, '
       'bad anatomy, blurry, lowres, jpeg artifacts, chibi, sketch, nudity, revealing clothing, '
       'suggestive, child, swastika')
NEG_BG = NEG + ', people, crowd, face, portrait'


def scripts():
    return sorted(glob.glob(os.path.join(ROOT, 'script', 'ch*.txt')))


def shots(upto=None):
    """(id, prompt, chapter) for every [img] cue in every script, first occurrence only.

    `upto` limits it to chapters already rewritten, so generation can run alongside writing.
    """
    out, seen = [], set()
    for path in scripts():
        ch = int(re.search(r'ch(\d+)\.txt$', path).group(1))
        if upto and ch > upto:
            continue
        for line in open(path):
            s = line.strip()
            if not s.startswith('[img]'):
                continue
            prompt = s[len('[img]'):].strip()
            sid = f'c{ch:02d}_' + hashlib.sha1(prompt.encode()).hexdigest()[:10]
            if sid not in seen:
                seen.add(sid)
                out.append((sid, prompt, ch))
    return out


def prepare(prompt, look):
    """A script's [img] line as the model should see it, with its negative after ' ### '."""
    tags = re.findall(r'\{([A-Z_]+)\}', prompt)
    people = [t for t in tags if t not in ('UMBRELLA',)]
    p = re.sub(r'\{([A-Z_]+)\}', lambda m: look.get(m.group(1), m.group(1).lower()), prompt)
    p = re.sub(r',?\s*\bcinematic( wide shot| close-up)?\b', '', p).strip(' ,')
    for pat, rep in gen_images.MARK:
        p = re.sub(pat, rep, p)
    p = re.sub(r'\s{2,}', ' ', re.sub(r'(,\s*)+', ', ', p)).strip(' ,')
    if people:
        # tag-trained models need the head count stated, or every shot becomes a solo portrait
        return f'{count_tag(people, look)}, {p}, {CHAR_TAIL} ### {NEG}'
    return f'{p}, {BG_TAIL} ### {NEG_BG}'


def count_tag(people, look):
    if len(people) > 2:
        return 'multiple people, group'
    men = sum(1 for t in people if re.search(r'\b(man|boy|male|soldier|officer|general|admiral)\b',
                                             look.get(t, '')))
    women = len(people) - men
    bits = []
    if men:
        bits.append('1boy' if men == 1 else f'{men}boys')
    if women:
        bits.append('1girl' if women == 1 else f'{women}girls')
    return ', '.join(bits) + (', solo' if len(people) == 1 else '')


def finish(img, seed):
    """1344x768 -> 1344x756 (16:9) -> 1920x1080, with a light sharpen. No film grain: the
    grain that suited photographs reads as dirt on flat cel shading."""
    from PIL import ImageFilter
    img = img.crop((0, 6, W, 762)).resize((1920, 1080), resample=3)
    return img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=30, threshold=3))


def main():
    os.makedirs(OUT, exist_ok=True)
    look = {k: v for k, v in json.load(open(os.path.join(ROOT, 'story', 'visuals.json'))).items()
            if not k.startswith('_')}
    upto = int(sys.argv[sys.argv.index('--upto') + 1]) if '--upto' in sys.argv else None
    all_shots = shots(upto)
    if '--report' in sys.argv:
        for sid, p, ch in all_shots:
            print(f'{sid}\t{prepare(p, look)}')
        print(f'{len(all_shots)} shots', file=sys.stderr)
        return
    window = int(sys.argv[sys.argv.index('--inflight') + 1]) if '--inflight' in sys.argv else 24
    limit = int(sys.argv[sys.argv.index('--limit') + 1]) if '--limit' in sys.argv else None

    jobs = deque()
    for sid, p, ch in all_shots:
        if not os.path.exists(os.path.join(OUT, sid + '.jpg')):
            jobs.append({'name': sid, 'prompt': prepare(p, look), 'tries': 0, 'model': 'anime',
                         'seed': int(hashlib.sha1(sid.encode()).hexdigest()[:8], 16)})
    tag = 'rev' if '--reverse' in sys.argv else 'fwd'
    by_name = {j['name']: j for j in jobs}
    ledger_path = os.path.join(OUT, f'inflight_{tag}.tsv')
    adopted = {}
    if os.path.exists(ledger_path):
        for line in open(ledger_path):
            try:
                rid, name, model, seed, ts = line.rstrip('\n').split('\t')
            except ValueError:
                continue
            j = by_name.get(name)
            if j and time.time() - float(ts) < 2400 and not os.path.exists(os.path.join(OUT, name + '.jpg')):
                j.update(model=model, seed=int(seed), t=float(ts))
                adopted[rid] = j
    if adopted:
        names = {j['name'] for j in adopted.values()}
        jobs = deque(j for j in jobs if j['name'] not in names)
        print(f'adopted {len(adopted)} requests still in flight', flush=True)
    if '--reverse' in sys.argv:
        jobs = deque(reversed(jobs))
    if limit:
        jobs = deque(list(jobs)[:limit])
    total = len(all_shots)
    print(f'{len(jobs)} frames to generate of {total}; {window} in flight', flush=True)

    saves = queue.Queue()
    done = {'n': 0}
    lock = threading.Lock()
    log = open(os.path.join(OUT, 'prompts.tsv'), 'a')
    fails = open(os.path.join(OUT, 'failures.txt'), 'a')

    def saver():
        while True:
            item = saves.get()
            if item is None:
                return
            job, img, worker = item
            try:
                out = finish(img, job['seed'])
                tmp = os.path.join(OUT, '.' + job['name'] + '.jpg')
                out.save(tmp, quality=92, subsampling=1, optimize=True)
                os.replace(tmp, os.path.join(OUT, job['name'] + '.jpg'))
                with lock:
                    log.write(f"{job['name']}\t{job['model']}\t{job['seed']}\t{worker}\t{job['prompt']}\n")
                    log.flush()
                    done['n'] += 1
            except Exception as e:
                print(f"  save failed {job['name']}: {e}", flush=True)
            finally:
                saves.task_done()

    savers = [threading.Thread(target=saver, daemon=True) for _ in range(2)]
    for t in savers:
        t.start()

    inflight = dict(adopted)
    ledger = open(ledger_path, 'a')

    def cancel_all(*_):
        for rid in list(inflight):
            try:
                horde._req('DELETE', f'/generate/status/{rid}', timeout=10)
            except Exception:
                pass
        print(f'stopped; withdrew {len(inflight)} queued requests', flush=True)
        os._exit(0)

    signal.signal(signal.SIGTERM, cancel_all)
    signal.signal(signal.SIGINT, cancel_all)

    def retry(job, why):
        job['tries'] += 1
        job['seed'] += 7919
        if job['tries'] == 2:
            job['model'] = 'anime2'
        if job['tries'] > 4:
            fails.write(f"{job['name']}\t{why}\n")
            fails.flush()
            print(f"  giving up on {job['name']} ({why})", flush=True)
        else:
            jobs.appendleft(job)

    t0 = time.time()
    last_report = 0
    backoff = 0
    while jobs or inflight:
        while jobs and len(inflight) < window and time.time() >= backoff:
            job = jobs.popleft()
            if os.path.exists(os.path.join(OUT, job['name'] + '.jpg')):
                continue
            try:
                rid = horde.submit(job['prompt'], job['seed'], job['model'], W, H, KEY)
                job['t'] = time.time()
                inflight[rid] = job
                ledger.write(f"{rid}\t{job['name']}\t{job['model']}\t{job['seed']}\t{job['t']}\n")
                ledger.flush()
                time.sleep(2.0)          # the anonymous key is rate limited on submission
            except urllib.error.HTTPError as e:
                body = e.read()[:200]
                if e.code in (429, 503):
                    jobs.appendleft(job)
                    backoff = time.time() + 30
                    print(f'  horde busy ({e.code}); backing off 30s', flush=True)
                else:
                    print(f"  submit rejected {e.code}: {body}", flush=True)
                    retry(job, f'submit {e.code}')
                break
            except Exception as e:
                jobs.appendleft(job)
                backoff = time.time() + 15
                print(f'  submit error {type(e).__name__}; retrying in 15s', flush=True)
                break
        for rid in list(inflight):
            job = inflight[rid]
            try:
                c = horde.check(rid)
            except Exception:
                time.sleep(1)
                continue
            if c.get('faulted'):
                del inflight[rid]
                retry(job, 'faulted')
            elif c.get('done'):
                del inflight[rid]
                try:
                    img, meta = horde.fetch(rid)
                except Exception as e:
                    img, meta = None, {'err': str(e)}
                if img is None:
                    retry(job, 'censored' if isinstance(meta, dict) and meta.get('censored') else 'empty')
                else:
                    saves.put((job, img, meta.get('worker_name', '') if isinstance(meta, dict) else ''))
            elif not c.get('is_possible', True) or time.time() - job['t'] > 2400:
                del inflight[rid]
                try:
                    horde._req('DELETE', f'/generate/status/{rid}')
                except Exception:
                    pass
                job['model'] = 'anime2' if job['model'] == 'anime' else 'anime'
                jobs.appendleft(job)
            time.sleep(0.4)
        if time.time() - last_report > 180:
            last_report = time.time()
            el = time.time() - t0
            on_disk = len([f for f in os.listdir(OUT) if f.endswith('.jpg') and not f.startswith('.')])
            rate = done['n'] / el * 3600 if el > 0 else 0
            print(f'[{time.strftime("%H:%M:%S")}] {tag}: {done["n"]} saved here, {on_disk}/{total} on disk, '
                  f'{len(inflight)} in flight, {rate:.0f}/h, {total - on_disk} left', flush=True)
        time.sleep(2)

    saves.join()
    for _ in savers:
        saves.put(None)
    print(f'DONE: {done["n"]} saved in {(time.time() - t0) / 3600:.2f}h', flush=True)


if __name__ == '__main__':
    main()
