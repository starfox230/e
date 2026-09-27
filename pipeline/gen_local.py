"""Generate scene art locally, on the CPU, for whatever the Horde has not got to.

The Horde's anonymous queue is the better picture and the worse schedule: AAM XL draws a
frame this film can use, but anonymous requests sit behind every registered user's, and a
queue position of two hundred is normal. This fills in behind it.

Anything-V5 (Stable Diffusion 1.5) with the LCM LoRA fused in runs in eight steps instead of
twenty-five, which is the difference between four minutes a frame and seventy seconds on four
cores. The result is flatter and looser in the hands than the Horde's, so this is the
fallback: it works the shot list from the middle outwards, skips anything already on disk,
and the Horde's frames win wherever both exist.

Usage: python3 pipeline/gen_local.py [--upto N] [--limit N]
"""
import os
import sys
import time
import hashlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_anime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get('GEN_OUT') or os.path.join(ROOT, 'work', 'images_anime')
MODEL = os.path.join(ROOT, 'work', 'models', 'anything-v5')
LCM = os.path.join(ROOT, 'work', 'models', 'lcm-lora-sdv1-5')
W, H = 768, 448          # 1.71:1; cropped to 768x432 and taken to 1920x1080
STEPS = 8


def load():
    import torch
    from diffusers import StableDiffusionPipeline, LCMScheduler
    torch.set_num_threads(os.cpu_count() or 4)
    pipe = StableDiffusionPipeline.from_pretrained(MODEL, safety_checker=None,
                                                   requires_safety_checker=False)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(LCM)
    pipe.fuse_lora()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def finish(img):
    from PIL import ImageFilter
    img = img.crop((0, 8, W, 440)).resize((1920, 1080), resample=3)
    return img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=35, threshold=3))


def order(shots):
    """Middle outwards: the two Horde processes work in from the ends."""
    mid = len(shots) // 2
    return sorted(shots, key=lambda s: abs(shots.index(s) - mid))


def main():
    import torch
    upto = int(sys.argv[sys.argv.index('--upto') + 1]) if '--upto' in sys.argv else None
    limit = int(sys.argv[sys.argv.index('--limit') + 1]) if '--limit' in sys.argv else None
    look = gen_anime.json.load(open(os.path.join(ROOT, 'story', 'visuals.json')))
    look = {k: v for k, v in look.items() if not k.startswith('_')}
    shots = order(gen_anime.shots(upto))
    if limit:
        shots = shots[:limit]
    os.makedirs(OUT, exist_ok=True)
    print(f'{len(shots)} shots in the list; loading the model', flush=True)
    pipe = load()
    done = 0
    t0 = time.time()
    for sid, prompt, ch in shots:
        path = os.path.join(OUT, sid + '.jpg')
        if os.path.exists(path):
            continue
        full = gen_anime.prepare(prompt, look)
        pos, neg = full.split(' ### ', 1)
        seed = int(hashlib.sha1(sid.encode()).hexdigest()[:8], 16) % (2 ** 31)
        t = time.time()
        img = pipe(pos, negative_prompt=neg, num_inference_steps=STEPS, guidance_scale=1.6,
                   width=W, height=H, generator=torch.Generator().manual_seed(seed)).images[0]
        if os.path.exists(path):
            continue                      # the Horde got there while this one was rendering
        tmp = os.path.join(OUT, '.' + sid + '.jpg')
        finish(img).save(tmp, quality=92, subsampling=1, optimize=True)
        os.replace(tmp, path)
        done += 1
        on_disk = len([f for f in os.listdir(OUT) if f.endswith('.jpg') and not f.startswith('.')])
        print(f'[{time.strftime("%H:%M:%S")}] {sid} {time.time() - t:.0f}s  '
              f'{done} here, {on_disk}/{len(shots)} on disk', flush=True)
    print(f'DONE: {done} in {(time.time() - t0) / 3600:.2f}h', flush=True)


if __name__ == '__main__':
    main()
