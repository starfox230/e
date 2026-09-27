"""Generate the film's scene art with FLUX.1-schnell on AI Horde.

Side by side on the same scenes, FLUX beat the local one-step SDXL-Turbo on everything the
story needs: "Wesker looking at Nadia across a desk" came back as two people at a desk
rather than one, the Tyrant came back as the Tyrant rather than a gunman, and faces read
as photographs rather than wax. The Horde runs it on volunteers' GPUs for free.

Frames are requested at 1344x768, cropped to 1344x756 (16:9) and taken to 1920x1080 with a
plain Lanczos resize and a light sharpen -- 1.43x needs no super-resolution model, and the
one used for the local frames smoothed skin toward an airbrushed finish.

The Horde is a volunteer network and anonymous requests share one low-priority account, so
the client keeps a modest number of requests in flight rather than flooding it, and polls
gently. A request that faults, or is caught by the Horde's content filter, is retried with
a fresh seed; if FLUX workers disappear it falls back to Z-Image-Turbo. Anything that still
fails is listed in failures.txt for the local generator to fill.

Prompts go through gen_images.prepare(), the same rewriting the local generator used.

Usage: python3 pipeline/gen_horde.py [--limit N] [--inflight N]
       HORDE_KEY=<key> python3 pipeline/gen_horde.py     (a registered key queues ahead of anonymous)
"""
import os
import sys
import time
import queue
import threading
import urllib.error
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import horde
import gen_images

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get('GEN_OUT') or os.path.join(ROOT, 'work', 'images_flux')
W, H = 1344, 768
KEY = os.environ.get('HORDE_KEY') or horde.ANON


def finish(img, seed):
    """1344x768 -> 1344x756 (16:9) -> 1920x1080, then the film's grain and vignette."""
    from PIL import ImageFilter
    img = img.crop((0, 6, W, 762)).resize((1920, 1080), resample=3)     # 3 = LANCZOS
    img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=40, threshold=2))
    return gen_images.grade(img, seed)


def main():
    os.makedirs(OUT, exist_ok=True)
    limit = int(sys.argv[sys.argv.index('--limit') + 1]) if '--limit' in sys.argv else None
    window = int(sys.argv[sys.argv.index('--inflight') + 1]) if '--inflight' in sys.argv else 20
    look = gen_images.looks()
    jobs = deque()
    total = 0
    for sid, p, yr in gen_images.shots():
        for v in ('', '_v1'):
            total += 1
            name = sid + v
            if not os.path.exists(os.path.join(OUT, name + '.jpg')):
                jobs.append({'name': name, 'prompt': gen_images.prepare(p, None, look, yr),
                             'seed': gen_images.seed_of(name), 'tries': 0, 'model': 'flux'})
    if limit:
        jobs = deque(list(jobs)[:limit])
    n_jobs = len(jobs)
    print(f'{n_jobs} frames to generate ({total - n_jobs} of {total} already on disk); '
          f'{window} in flight, {"registered" if KEY != horde.ANON else "anonymous"} key', flush=True)

    saves = queue.Queue()
    done = {'n': 0}
    lock = threading.Lock()
    promptlog = open(os.path.join(OUT, 'prompts.tsv'), 'a')
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
                    promptlog.write(f"{job['name']}\t{job['model']}\t{job['seed']}\t{worker}\t{job['prompt']}\n")
                    promptlog.flush()
                    done['n'] += 1
            except Exception as e:
                print(f"  save failed {job['name']}: {e}", flush=True)
            finally:
                saves.task_done()

    savers = [threading.Thread(target=saver, daemon=True) for _ in range(2)]
    for t in savers:
        t.start()

    inflight = {}                  # request id -> job
    t0 = time.time()
    last_report = 0
    backoff = 0

    def retry(job, why):
        job['tries'] += 1
        job['seed'] += 7919       # a fresh composition, still reproducible
        if job['tries'] >= 2 and job['model'] == 'flux':
            job['model'] = 'zimage'
        if job['tries'] > 4:
            fails.write(f"{job['name']}\t{why}\n")
            fails.flush()
            print(f"  giving up on {job['name']} ({why}); left for the local generator", flush=True)
        else:
            jobs.appendleft(job)

    while jobs or inflight:
        # top up the window
        while jobs and len(inflight) < window and time.time() >= backoff:
            job = jobs.popleft()
            try:
                rid = horde.submit(job['prompt'], job['seed'], job['model'], W, H, KEY)
                job['t'] = time.time()
                inflight[rid] = job
            except urllib.error.HTTPError as e:
                body = e.read()[:200]
                jobs.appendleft(job)
                if e.code in (429, 503):
                    backoff = time.time() + 30
                    print(f'  horde busy ({e.code}); backing off 30s', flush=True)
                else:
                    print(f"  submit rejected {e.code}: {body}", flush=True)
                    jobs.popleft()
                    retry(job, f'submit {e.code}')
                break
            except Exception as e:
                jobs.appendleft(job)
                backoff = time.time() + 15
                print(f'  submit error {type(e).__name__}; retrying in 15s', flush=True)
                break
        # poll what is in flight, gently
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
            elif not c.get('is_possible', True) or time.time() - job['t'] > 1500:
                # no worker can take it, or it has sat far longer than any queue should
                del inflight[rid]
                try:
                    horde._req('DELETE', f'/generate/status/{rid}')
                except Exception:
                    pass
                job['tries'] += 1
                job['model'] = 'zimage' if job['model'] == 'flux' else 'flux'
                jobs.appendleft(job)
            time.sleep(0.4)
        if time.time() - last_report > 120:
            last_report = time.time()
            el = time.time() - t0
            n = done['n']
            rate = n / el * 3600 if el > 0 else 0
            left = n_jobs - n
            eta = left / rate if rate > 0 else float('inf')
            print(f'[{time.strftime("%H:%M:%S")}] {n}/{n_jobs} saved, {len(inflight)} in flight, '
                  f'{rate:.0f}/h, eta {eta:.1f}h', flush=True)
        time.sleep(2)

    saves.join()
    for _ in savers:
        saves.put(None)
    print(f'DONE: {done["n"]} saved in {(time.time() - t0) / 3600:.2f}h', flush=True)


if __name__ == '__main__':
    main()
