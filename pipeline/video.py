"""Render a chapter's picture from its timeline and mux it with the mixed soundtrack.

Frames are composed with OpenCV (sub-pixel Ken Burns via warpAffine, crossfades, overlays)
and piped raw into ffmpeg. Every frame time comes from the same timeline the audio used.

Usage: python3 pipeline/video.py 1 [--preview SECONDS]
"""
import os
import sys
import json
import math
import hashlib
import subprocess
import numpy as np
import cv2
from PIL import Image
import graphics as G

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The rebuilt film's frames are the anime ones; work/images holds the old procedural art and
# is the fallback for any shot the generators have not reached yet. IMAGES can be overridden
# for a test render.
IMAGES = os.environ.get('FILM_IMAGES') or (
    os.path.join(ROOT, 'work', 'images_anime')
    if os.path.isdir(os.path.join(ROOT, 'work', 'images_anime'))
    else os.path.join(ROOT, 'work', 'images'))
FALLBACK_IMAGES = os.path.join(ROOT, 'work', 'images')
W, H, FPS = 1920, 1080, 24
XFADE = 1.1          # crossfade between shots (s)
SUBSHOT = 11.0       # a long shot is re-framed every ~11 s so the picture never goes stale
PANEL_IN = 0.35


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


class Source:
    """A still prepared for camera moves: resized to cover 1920x1080 with head-room for zoom."""
    cache = {}

    @classmethod
    def get(cls, path, grade):
        key = (path, grade)
        if key not in cls.cache:
            if len(cls.cache) > 6:
                cls.cache.pop(next(iter(cls.cache)))
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(path)
            h, w = img.shape[:2]
            s = max(W * 1.2 / w, H * 1.2 / h)
            img = cv2.resize(img, (int(w * s + 0.5), int(h * s + 0.5)), interpolation=cv2.INTER_LANCZOS4)
            cls.cache[key] = color_grade(img, grade)
        return cls.cache[key]


def color_grade(img, grade):
    f = img.astype(np.float32) / 255.0
    if grade == 'earth':        # teal shadows, warm highlights, slight desaturation
        lum = f.mean(axis=2, keepdims=True)
        f = lum + (f - lum) * 0.88
        f[..., 0] += (0.5 - lum[..., 0]) * 0.05     # B up in shadows
        f[..., 2] += (lum[..., 0] - 0.5) * 0.05     # R up in highlights
    elif grade == 'void':       # cold, deep blacks
        f = f ** 1.08
        f[..., 0] *= 1.04
        f[..., 2] *= 0.95
    elif grade == 'blood':
        lum = f.mean(axis=2, keepdims=True)
        f = lum + (f - lum) * 0.8
        f[..., 2] *= 1.05
    f = np.clip((f - 0.02) / 0.98, 0, 1)            # lift-free contrast
    return (f * 255).astype(np.uint8)


def motion_for(seed, dur):
    """Deterministic camera move for a (sub)shot: returns function t->(scale, cx, cy) in normalized coords."""
    r = np.random.default_rng(int(hashlib.md5(seed.encode()).hexdigest()[:8], 16))
    kind = r.choice(['in', 'out', 'pan_lr', 'pan_rl', 'in', 'out', 'drift'])
    z0, z1 = 1.0, 1.0 + r.uniform(0.07, 0.12) * min(1.0, dur / 10.0 + 0.3)
    fx0, fy0 = r.uniform(-0.25, 0.25), r.uniform(-0.2, 0.2)
    fx1, fy1 = fx0 + r.uniform(-0.3, 0.3), fy0 + r.uniform(-0.15, 0.15)
    if kind == 'out':
        z0, z1 = z1, z0
    if kind in ('pan_lr', 'pan_rl'):
        z0 = z1 = 1.08
        fx0, fx1 = (-0.7, 0.7) if kind == 'pan_lr' else (0.7, -0.7)
        fy0 = fy1 = r.uniform(-0.2, 0.2)
    if kind == 'drift':
        z0, z1 = 1.05, 1.1

    def at(t):
        u = ease(t / max(dur, 1e-3)) * 0.85 + (t / max(dur, 1e-3)) * 0.15
        return z0 + (z1 - z0) * u, fx0 + (fx1 - fx0) * u, fy0 + (fy1 - fy0) * u
    return at


def render_still(src, scale, fx, fy):
    """Crop a W x H view out of src at zoom `scale` (1 = cover) centered by fx, fy in [-1, 1].

    Only the visible region is touched: the crop is taken with a one-pixel margin and the
    sub-pixel remainder is applied by warping that small patch, which is far cheaper than
    warping the whole oversized source every frame.
    """
    sh, sw = src.shape[:2]
    base = max(W / sw, H / sh)
    s = base * scale
    vw, vh = W / s, H / s                     # view size in source pixels
    cx = sw / 2 + fx * (sw - vw) / 2
    cy = sh / 2 + fy * (sh - vh) / 2
    x0f, y0f = cx - vw / 2, cy - vh / 2
    x0, y0 = int(math.floor(x0f)), int(math.floor(y0f))
    fracx, fracy = x0f - x0, y0f - y0
    x1, y1 = x0 + int(math.ceil(vw)) + 2, y0 + int(math.ceil(vh)) + 2
    # clamp, remembering how much we had to shift so the framing stays correct
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(sw, x1), min(sh, y1)
    patch = src[cy0:cy1, cx0:cx1]
    if patch.size == 0:
        return np.zeros((H, W, 3), np.uint8)
    off_x = fracx + (x0 - cx0)
    off_y = fracy + (y0 - cy0)
    M = np.array([[s, 0, -off_x * s], [0, s, -off_y * s]], np.float32)
    return cv2.warpAffine(patch, M, (W, H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


class Overlay:
    def __init__(self, rgba):
        a = np.array(rgba)
        self.rgb = cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2BGR).astype(np.float32)
        self.alpha = (a[..., 3:4].astype(np.float32) / 255.0)
        self.h, self.w = a.shape[:2]


def blit(frame, ov, x, y, opacity=1.0, mask=None):
    x, y = int(x), int(y)
    h, w = ov.h, ov.w
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sub = frame[y0:y1, x0:x1].astype(np.float32)
    a = ov.alpha[y0 - y:y1 - y, x0 - x:x1 - x] * opacity
    if mask is not None:
        a = a * mask[y0 - y:y1 - y, x0 - x:x1 - x]
    frame[y0:y1, x0:x1] = (sub * (1 - a) + ov.rgb[y0 - y:y1 - y, x0 - x:x1 - x] * a).astype(np.uint8)


def reveal_mask(shape, boxes, frac):
    """Type-on reveal: rows appear one after another, left to right."""
    m = np.zeros(shape[:2] + (1,), np.float32)
    total = sum(max(1, b[2] - b[0]) for b in boxes)
    budget = frac * total
    for x0, y0, x1, y1 in boxes:
        wdt = max(1, x1 - x0)
        if budget <= 0:
            break
        take = min(wdt, budget)
        m[y0:y1, x0:int(x0 + take)] = 1.0
        budget -= take
    return m


def grade_for(chapter):
    # the rebuilt story leaves the planet at chapter 42
    if chapter >= 42:
        return 'void'
    return 'earth'


def image_path(shot):
    for d in (IMAGES, FALLBACK_IMAGES):
        for ext in ('jpg', 'png', 'webp'):
            p = os.path.join(d, f"{shot['id']}.{ext}")
            if os.path.exists(p):
                return p
    # placeholder so the pipeline can run before images exist
    os.makedirs(os.path.join(IMAGES, 'placeholder'), exist_ok=True)
    p = os.path.join(IMAGES, 'placeholder', f"{shot['id']}.png")
    if not os.path.exists(p):
        G.placeholder(shot['prompt'], seed=int(shot['id'][-6:], 16)).save(p)
    return p


def variants_of(shot):
    """Every rendered composition of a shot, base first."""
    out = [image_path(shot)]
    k = 1
    while True:
        found = None
        for ext in ('jpg', 'png', 'webp'):
            p = os.path.join(IMAGES, f"{shot['id']}_v{k}.{ext}")
            if os.path.exists(p):
                found = p
                break
        if not found:
            break
        out.append(found)
        k += 1
    return out


def punch_in(seed, dur):
    """A tighter framing of the same photograph for the next beat of a long shot: the move a
    documentary editor makes when a still has to hold for a minute."""
    r = np.random.default_rng(int(hashlib.md5(seed.encode()).hexdigest()[:8], 16))
    z0 = r.uniform(1.24, 1.32)
    z1 = z0 + r.uniform(0.04, 0.07) * (1 if r.random() < 0.7 else -1)
    fx0, fy0 = r.uniform(-0.35, 0.35), r.uniform(-0.3, 0.1)        # subjects sit centre-high
    fx1, fy1 = fx0 + r.uniform(-0.12, 0.12), fy0 + r.uniform(-0.06, 0.06)

    def at(t):
        u = ease(t / max(dur, 1e-3)) * 0.85 + (t / max(dur, 1e-3)) * 0.15
        return z0 + (z1 - z0) * u, fx0 + (fx1 - fx0) * u, fy0 + (fy1 - fy0) * u
    return at


def build_plan(tl):
    """Split shots into sub-shots and build per-sub-shot camera moves.

    A shot is one photograph for its whole length. Long shots are broken into beats that cut
    between framings of it -- wide, then a tighter punch-in, then wide again -- rather than
    dissolving to a second generated take: two photographs of "the same" building or person
    are two different buildings and people, and cutting between them mid-shot reads as a
    continuity error. Crossfades are kept for the joins between shots, where the picture
    really does change."""
    subs = []
    for s in tl['shots']:
        dur = s['end'] - s['start']
        path = image_path(s)
        n = max(1, int(round(dur / SUBSHOT)))
        seg = dur / n
        for k in range(n):
            st = s['start'] + k * seg
            move = (punch_in if k % 2 else motion_for)(f"{s['id']}_{k}", seg + XFADE)
            subs.append({'start': st, 'end': st + seg, 'path': path, 'id': s['id'],
                         'move': move, 'first': k == 0, 'cut': k > 0})
    return subs


class Composer:
    def __init__(self, chapter):
        self.chapter = chapter
        self.work = os.path.join(ROOT, 'work', f'ch{chapter:02d}')
        tl = json.load(open(os.path.join(self.work, 'timeline.json')))
        self.tl = tl
        self.dur = tl['duration']
        self.grade = grade_for(chapter)
        self.subs = build_plan(tl)
        self.panels = []
        for p in tl['system']:
            img, boxes, txt = G.system_panel(p['title'], p['lines'])
            self.panels.append({**p, 'bg': Overlay(img), 'txt': Overlay(txt), 'boxes': boxes})
        self.locs = []
        for l in tl['loc']:
            img, box = G.loc_caption(l['text'])
            self.locs.append({**l, 'ov': Overlay(img), 'box': box})
        self.cards = []
        for c in tl['cards']:
            if c['type'] == 'movement':
                im = G.movement_card(c['num'], c['title'])
            elif c['type'] == 'chapter':
                im = G.chapter_card(c['num'], c['title'], c.get('movement'))
            else:
                im = G.main_title()
            self.cards.append({**c, 'img': cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)})
        yy, xx = np.mgrid[0:H, 0:W]
        vig = 1 - 0.38 * np.clip(np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) - 0.35, 0, 1) ** 1.5
        self.vig = (np.dstack([vig] * 3) * 255).astype(np.uint8)
        self.si = 0

    def sub_index(self, t):
        subs = self.subs
        if self.si >= len(subs) or subs[self.si]['start'] > t:
            self.si = 0
        while self.si + 1 < len(subs) and subs[self.si + 1]['start'] <= t:
            self.si += 1
        return self.si

    def frame(self, t, end=None):
        end = end if end is not None else self.dur
        subs = self.subs
        if subs:
            si = self.sub_index(t)
            cur = subs[si]
            src = Source.get(cur['path'], self.grade)
            frame = render_still(src, *cur['move'](t - cur['start']))
            if si > 0 and t - cur['start'] < XFADE and not cur.get('cut'):
                prev = subs[si - 1]
                psrc = Source.get(prev['path'], self.grade)
                pframe = render_still(psrc, *prev['move'](t - prev['start']))
                a = ease((t - cur['start']) / XFADE)
                frame = cv2.addWeighted(frame, a, pframe, 1 - a, 0)
        else:
            frame = np.zeros((H, W, 3), np.uint8)

        for p in self.panels:
            if p['start'] - 0.05 <= t <= p['end'] + 0.4:
                k_in = ease((t - p['start']) / PANEL_IN)
                k_out = 1 - ease((t - p['end']) / 0.4) if t > p['end'] else 1.0
                vis = k_in * k_out
                small = cv2.resize(frame, (W // 6, H // 6), interpolation=cv2.INTER_AREA)
                small = cv2.GaussianBlur(small, (0, 0), 3)
                blur = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
                dimmed = cv2.addWeighted(blur, 0.55, np.zeros_like(blur), 0.45, 0)
                frame = cv2.addWeighted(dimmed, vis * 0.85, frame, 1 - vis * 0.85, 0)
                x = (W - p['bg'].w) / 2
                y = (H - p['bg'].h) / 2 + (1 - k_in) * 30
                blit(frame, p['bg'], x, y, vis)
                type_frac = ease((t - p['start'] - 0.25) / max(0.8, min(2.2, 0.24 * len(p['boxes']))))
                mask = reveal_mask((p['txt'].h, p['txt'].w), p['boxes'], type_frac)
                blit(frame, p['txt'], x, y, vis, mask)

        for l in self.locs:
            if l['start'] <= t <= l['end'] + 0.6:
                vis = 1 - ease((t - l['end']) / 0.6) if t > l['end'] else 1.0
                frac = ease((t - l['start']) / 1.4)
                x0, y0, x1, y1 = l['box']
                mask = np.zeros((l['ov'].h, l['ov'].w, 1), np.float32)
                mask[:, :x0] = 1.0
                mask[:, x0:int(x0 + (x1 - x0) * frac) + 1] = 1.0
                blit(frame, l['ov'], 90, H - 190, vis, mask)

        for c in self.cards:
            if c['start'] - 0.8 <= t <= c['end'] + 0.8:
                if t < c['start']:
                    a = ease((t - c['start'] + 0.8) / 0.8)
                elif t > c['end']:
                    a = 1 - ease((t - c['end']) / 0.8)
                else:
                    a = 1.0
                z = 1.0 + 0.04 * (t - c['start']) / max(1.0, c['end'] - c['start'])
                M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, z)
                card = cv2.warpAffine(c['img'], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                frame = cv2.addWeighted(card, a, frame, 1 - a, 0)

        if t < 1.0:
            frame = cv2.multiply(frame, np.full_like(frame, int(255 * ease(t / 1.0))), scale=1 / 255)
        if t > end - 1.5:
            frame = cv2.multiply(frame, np.full_like(frame, int(255 * ease((end - t) / 1.5))), scale=1 / 255)
        return frame


def render(chapter, preview=None, out=None, start=0.0):
    comp = Composer(chapter)
    work = comp.work
    dur = comp.dur if preview is None else min(preview, comp.dur - start)
    end = start + dur
    nframes = int(math.ceil(dur * FPS))
    out = out or os.path.join(work, 'video.mp4' if preview is None else 'preview.mp4')
    audio = os.path.join(work, 'mix.wav')
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS),
           '-i', '-', '-ss', f'{start:.3f}', '-i', audio, '-t', f'{dur:.3f}', '-map', '0:v', '-map', '1:a',
           '-c:v', 'libx264', '-preset', os.environ.get('X264_PRESET', 'medium'), '-crf', os.environ.get('X264_CRF', '21'),
           '-tune', 'film', '-pix_fmt', 'yuv420p', '-g', str(FPS * 4), '-c:a', 'aac', '-b:a', '192k',
           '-movflags', '+faststart', out]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in range(nframes):
        t = start + fi / FPS
        frame = comp.frame(t, end=comp.dur if preview is None else end + 10)
        ff.stdin.write(frame.tobytes())
        if fi % (FPS * 60) == 0:
            print(f'  frame {fi}/{nframes} ({100 * fi / nframes:.0f}%)', flush=True)
    ff.stdin.close()
    ff.wait()
    if ff.returncode:
        raise RuntimeError('ffmpeg failed')
    return out


def stills(chapter, times, outdir):
    comp = Composer(chapter)
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for t in times:
        f = comp.frame(t)
        p = os.path.join(outdir, f'ch{chapter:02d}_{t:08.2f}.jpg')
        cv2.imwrite(p, cv2.resize(f, (960, 540), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])
        paths.append(p)
    return paths


if __name__ == '__main__':
    ch = int(sys.argv[1])
    if '--stills' in sys.argv:
        ts = [float(x) for x in sys.argv[sys.argv.index('--stills') + 1].split(',')]
        print('\n'.join(stills(ch, ts, '/tmp/claude-0/stills')))
        sys.exit()
    prev = float(sys.argv[sys.argv.index('--preview') + 1]) if '--preview' in sys.argv else None
    st = float(sys.argv[sys.argv.index('--start') + 1]) if '--start' in sys.argv else 0.0
    print(render(ch, prev, start=st))
