"""Procedural cinematic scene renderer.

No image model is reachable from this container, so scene art is composed from code: layered
silhouettes, atmospheric depth, volumetric light and film grain, driven by keywords in each
[img] prompt. Deterministic per shot id, so a shot always renders identically.

Usage:
    python3 pipeline/art.py --demo            # a sampler of scene types
    python3 pipeline/art.py --chapter 1       # every shot of a chapter
"""
import os
import re
import sys
import math
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops

W, H = 1920, 1080
# The brightest 1% of a graded frame must reach at least this, or the shot reads as black
# on screen. Kept low on purpose: the look is meant to be dark, just not featureless. Across
# the finished art the 5th percentile of this measure sits near 0.27, so a floor here catches
# the frames that genuinely failed without touching the ones that are merely moody.
FLOOR_P99 = 0.24
# Lifting to exactly the floor does not stick: the grain and the JPEG quantisation that come
# after cost a point or two, so the frame lands just under and gets caught again on the next
# pass forever. Aim above the floor and it clears in one go.
LIFT_TO_P99 = FLOOR_P99 * 1.2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'work', 'images')

# ----------------------------------------------------------------- palettes

PALETTES = {
    'night_rain':  [(8, 12, 20), (16, 26, 40), (36, 52, 72), (92, 116, 140), (196, 208, 220)],
    'night_city':  [(6, 8, 16), (18, 20, 36), (44, 44, 72), (120, 110, 140), (240, 210, 160)],
    'dawn':        [(18, 16, 28), (58, 44, 54), (140, 96, 84), (222, 156, 108), (255, 226, 178)],
    'day_grey':    [(38, 44, 52), (72, 82, 92), (124, 136, 148), (176, 186, 196), (228, 234, 240)],
    'office':      [(14, 16, 20), (34, 38, 46), (74, 80, 92), (150, 156, 170), (236, 240, 248)],
    'lab':         [(10, 16, 22), (22, 38, 50), (46, 80, 100), (130, 180, 200), (236, 248, 252)],
    'bunker':      [(10, 9, 8), (26, 22, 18), (56, 46, 36), (108, 92, 70), (198, 176, 140)],
    'fire':        [(12, 6, 4), (52, 16, 8), (140, 44, 12), (232, 108, 28), (255, 206, 120)],
    'blood':       [(10, 6, 8), (34, 10, 14), (92, 18, 26), (176, 40, 48), (248, 180, 170)],
    'space':       [(2, 3, 8), (8, 10, 22), (20, 26, 48), (70, 84, 130), (210, 222, 255)],
    'alien':       [(4, 8, 8), (12, 28, 30), (26, 62, 62), (86, 150, 140), (214, 248, 236)],
    'void':        [(2, 2, 4), (8, 6, 12), (22, 14, 26), (74, 40, 70), (208, 180, 220)],
    'sea':         [(8, 14, 20), (18, 34, 44), (44, 74, 88), (128, 170, 184), (238, 248, 250)],
    'gold':        [(14, 10, 6), (44, 30, 14), (108, 76, 28), (196, 148, 60), (252, 230, 170)],
    'clean':       [(24, 26, 30), (60, 64, 72), (128, 134, 146), (196, 202, 212), (246, 248, 252)],
    'desert':      [(20, 16, 14), (54, 42, 32), (120, 92, 66), (196, 156, 112), (246, 220, 176)],
    'snow':        [(20, 24, 30), (52, 60, 72), (110, 124, 142), (178, 192, 208), (244, 248, 252)],
}

CRIMSON = (196, 22, 36)


def rgb(c):
    return tuple(int(x) for x in c)


def lerp(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


class Scene:
    def __init__(self, prompt, seed):
        self.p = prompt.lower()
        self.rng = np.random.default_rng(seed)
        self.pal = PALETTES[self.pick_palette()]
        self.img = Image.new('RGB', (W, H), rgb(self.pal[0]))

    # ------------------------------------------------------------- analysis
    def has(self, *words):
        for w in words:
            w = w.strip()
            if ' ' in w or '-' in w:
                if w in self.p:
                    return True
            elif re.search(r'\b' + re.escape(w) + r'\b', self.p):
                return True
        return False

    def pick_palette(self):
        p = self.p
        if self.has('space', 'orbit', 'starfield', 'stars', 'void', 'interstellar', 'fleet', 'nebula', 'asteroid'):
            if self.has('alien', 'choir', 'veth', 'hive', 'myriad', 'cathedral-like'):
                return 'alien'
            return 'space'
        if self.has('volcano', 'lava', 'fire', 'burning', 'embers'):
            return 'fire'
        if self.has('bunker', '1945', 'wartime', 'ruins of a bombed'):
            return 'bunker'
        if self.has('laboratory', 'lab ', 'containment', 'cryo', 'microscope', 'white corridor', 'sublevel', 'clean room'):
            return 'lab'
        if self.has('desert', 'mojave', 'dust', 'sand'):
            return 'desert'
        if self.has('snow', 'winter', 'davos', 'moscow', 'blizzard'):
            return 'snow'
        if self.has('ocean city', 'seafront', 'promenade', 'sea wall', 'harbour', 'floating city', 'arcology', 'ocean'):
            return 'sea'
        if self.has('dawn', 'sunrise', 'golden light', 'morning light', 'gold'):
            return 'dawn'
        if self.has('rain', 'storm', 'wet', 'downpour'):
            return 'night_rain'
        if self.has('night', 'dark', 'midnight', '3 a.m', 'lit window'):
            return 'night_city'
        if self.has('blood', 'crimson', 'red light', 'emergency light'):
            return 'blood'
        if self.has('boardroom', 'office', 'desk', 'conference', 'newsroom', 'courtroom', 'chamber'):
            return 'office'
        if self.has('hospital', 'clinic', 'school', 'classroom', 'atrium', 'lobby'):
            return 'clean'
        if self.has('grey', 'overcast', 'cloudy'):
            return 'day_grey'
        return 'night_city'

    # -------------------------------------------------------------- helpers
    def grad(self, top, bottom, gamma=1.0, band=None):
        y0, y1 = band or (0, H)
        n = max(1, y1 - y0)
        t = (np.linspace(0, 1, n) ** gamma)[:, None]
        col = np.array(top)[None, :] * (1 - t) + np.array(bottom)[None, :] * t
        strip = np.repeat(col[:, None, :], W, axis=1).astype(np.uint8)
        self.img.paste(Image.fromarray(strip), (0, y0))

    def glow(self, x, y, r, color, strength=1.0):
        layer = Image.new('RGB', (W, H), (0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.ellipse([x - r, y - r, x + r, y + r], fill=rgb(color))
        layer = layer.filter(ImageFilter.GaussianBlur(r * 0.55))
        self.img = ImageChops.add(self.img, Image.eval(layer, lambda v: int(v * strength)))

    def shape(self, pts, color, blur=0.0, alpha=255):
        layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(layer).polygon(pts, fill=rgb(color) + (alpha,))
        if blur:
            layer = layer.filter(ImageFilter.GaussianBlur(blur))
        self.img = Image.alpha_composite(self.img.convert('RGBA'), layer).convert('RGB')

    def rect(self, box, color, blur=0.0, alpha=255):
        layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(layer).rectangle(box, fill=rgb(color) + (alpha,))
        if blur:
            layer = layer.filter(ImageFilter.GaussianBlur(blur))
        self.img = Image.alpha_composite(self.img.convert('RGBA'), layer).convert('RGB')

    def ellipse(self, box, color, blur=0.0, alpha=255):
        layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(layer).ellipse(box, fill=rgb(color) + (alpha,))
        if blur:
            layer = layer.filter(ImageFilter.GaussianBlur(blur))
        self.img = Image.alpha_composite(self.img.convert('RGBA'), layer).convert('RGB')

    # --------------------------------------------------------- backgrounds
    def sky(self):
        p = self.pal
        if self.has('space', 'orbit', 'starfield', 'stars', 'void', 'interstellar', 'nebula', 'asteroid', 'fleet'):
            self.grad(p[0], p[1], 1.0)
            self.stars()
            if self.rng.random() < 0.7:
                self.nebula()
            return
        top = lerp(p[1], p[3], 0.25)
        self.grad(top, p[0], 1.6)
        # light source
        if self.has('dawn', 'sunrise', 'sunset', 'dusk', 'golden'):
            self.glow(int(W * self.rng.uniform(0.25, 0.75)), int(H * 0.62), 520, lerp(p[3], p[4], 0.5), 0.55)
        elif self.has('night', 'dark', 'rain', 'midnight'):
            self.glow(int(W * self.rng.uniform(0.2, 0.8)), int(H * 0.18), 420, lerp(p[2], p[3], 0.4), 0.3)

    def stars(self):
        a = np.array(self.img).astype(np.int16)
        r = self.rng
        n = 1400
        xs = r.integers(0, W, n)
        ys = r.integers(0, H, n)
        mag = (r.random(n) ** 3.2 * 255).astype(np.int16)
        for x, y, m in zip(xs, ys, mag):
            a[y, x] = np.minimum(255, a[y, x] + m)
            if m > 190:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < H and 0 <= xx < W:
                        a[yy, xx] = np.minimum(255, a[yy, xx] + m // 3)
        self.img = Image.fromarray(a.astype(np.uint8))

    def nebula(self):
        r = self.rng
        layer = Image.new('RGB', (W, H), (0, 0, 0))
        d = ImageDraw.Draw(layer)
        base = self.pal[2] if self.pal is not PALETTES['alien'] else self.pal[3]
        for _ in range(7):
            cx, cy = r.integers(0, W), r.integers(0, H)
            rr = r.integers(240, 700)
            c = lerp(base, self.pal[3], r.random())
            d.ellipse([cx - rr, cy - rr * 0.6, cx + rr, cy + rr * 0.6], fill=rgb(c))
        layer = layer.filter(ImageFilter.GaussianBlur(190))
        self.img = ImageChops.add(self.img, Image.eval(layer, lambda v: int(v * 0.30)))

    def horizon_haze(self, y):
        self.rect([0, y - 120, W, y + 40], lerp(self.pal[2], self.pal[3], 0.4), blur=90, alpha=90)

    # ---------------------------------------------------------- scene parts
    def skyline(self, y_base, depth=0, density=26, tall=False, lit=True):
        r = self.rng
        col = lerp(self.pal[0], self.pal[2], 0.12 + depth * 0.30)
        x = -r.integers(0, 120)
        while x < W + 100:
            w = int(r.integers(60, 190) * (1 + depth * 0.4))
            h = int(r.integers(90, 420) * (1.9 if tall else 1.0) * (1 - depth * 0.35))
            top = y_base - h
            self.rect([x, top, x + w, y_base], col)
            if lit and r.random() < 0.9:
                self.windows(x, top, w, h, depth)
            x += w + r.integers(6, 34)

    def windows(self, x, top, w, h, depth):
        r = self.rng
        layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        spill = Image.new('RGB', (W, H), (0, 0, 0))
        d = ImageDraw.Draw(layer)
        ds = ImageDraw.Draw(spill)
        warm = lerp(self.pal[4], (255, 190, 118), 0.55)
        cool = lerp(self.pal[3], (170, 210, 255), 0.5)
        pitch_x = max(9, int(13 - depth * 3))
        pitch_y = max(11, int(17 - depth * 4))
        cols = max(2, (w - 14) // pitch_x)
        rows = max(2, (h - 16) // pitch_y)
        ww = max(2, int(pitch_x * 0.55))
        wh = max(3, int(pitch_y * 0.5))
        lit_p = 0.30 - depth * 0.10
        for j in range(rows):
            wy = top + 10 + j * pitch_y
            floor_lit = r.random() < 0.82          # some floors are dark all the way across
            floor_tint = warm if r.random() < 0.85 else cool
            i = 0
            while i < cols:
                run = int(r.integers(1, 5))        # lit windows come in runs, not confetti
                lit = floor_lit and r.random() < lit_p * 2.4
                for k in range(run):
                    if i + k >= cols:
                        break
                    wx = x + 8 + (i + k) * pitch_x
                    if lit:
                        a = int(r.integers(140, 250))
                        d.rectangle([wx, wy, wx + ww, wy + wh], fill=rgb(floor_tint) + (a,))
                        if r.random() < 0.22:
                            ds.rectangle([wx - 1, wy - 1, wx + ww + 1, wy + wh + 1], fill=rgb(floor_tint))
                    else:
                        d.rectangle([wx, wy, wx + ww, wy + wh],
                                    fill=rgb(lerp(self.pal[0], self.pal[1], 0.78)) + (165,))
                i += run
        layer = layer.filter(ImageFilter.GaussianBlur(0.7))
        self.img = Image.alpha_composite(self.img.convert('RGBA'), layer).convert('RGB')
        self.img = ImageChops.add(self.img, Image.eval(spill.filter(ImageFilter.GaussianBlur(9)), lambda v: int(v * 0.45)))

    def brick_building(self, cx, y_base, w, h, sign=True):
        """The Wilmington building: four squat storeys, a faded sign, lit windows."""
        r = self.rng
        col = lerp(self.pal[0], (78, 50, 44), 0.42)
        self.rect([cx - w // 2, y_base - h, cx + w // 2, y_base], col)
        # parapet and cornice
        self.rect([cx - w // 2 - 8, y_base - h - 12, cx + w // 2 + 8, y_base - h + 6], lerp(col, self.pal[0], 0.35))
        # rooftop clutter
        for k in range(int(r.integers(2, 5))):
            bx = int(cx - w * 0.4 + r.random() * w * 0.8)
            bw, bh = int(r.integers(26, 70)), int(r.integers(16, 46))
            self.rect([bx, y_base - h - 12 - bh, bx + bw, y_base - h - 10], lerp(col, self.pal[0], 0.45))
        layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        spill = Image.new('RGB', (W, H), (0, 0, 0))
        d = ImageDraw.Draw(layer)
        ds = ImageDraw.Draw(spill)
        cols, rows = 9, 4
        pad = 22
        cw = (w - pad * 2) / cols
        rh = (h - 40) / rows
        for f in range(rows):
            for i in range(cols):
                wx = cx - w // 2 + pad + i * cw
                wy = y_base - h + 26 + f * rh
                on = r.random() < (0.55 if f == 2 else 0.20)
                c = lerp(self.pal[4], (255, 204, 138), 0.6) if on else lerp(self.pal[0], self.pal[1], 0.72)
                d.rectangle([wx, wy, wx + cw * 0.62, wy + rh * 0.52], fill=rgb(c) + (235 if on else 190,))
                if on:
                    ds.rectangle([wx - 2, wy - 2, wx + cw * 0.62 + 2, wy + rh * 0.52 + 2], fill=rgb(c))
        # doorway
        d.rectangle([cx - 26, y_base - 76, cx + 26, y_base], fill=rgb(lerp(self.pal[4], (255, 214, 150), 0.5)) + (200,))
        ds.rectangle([cx - 30, y_base - 80, cx + 30, y_base], fill=rgb(lerp(self.pal[4], (255, 214, 150), 0.5)))
        self.img = Image.alpha_composite(self.img.convert('RGBA'), layer.filter(ImageFilter.GaussianBlur(0.8))).convert('RGB')
        self.img = ImageChops.add(self.img, Image.eval(spill.filter(ImageFilter.GaussianBlur(14)), lambda v: int(v * 0.5)))
        if sign:
            sy = y_base - h - 14
            self.rect([cx - w // 5, sy - 30, cx + w // 5, sy], lerp(self.pal[2], (50, 84, 140), 0.55), alpha=225)
            self.glow(cx, sy - 15, 90, lerp(self.pal[3], (110, 150, 220), 0.5), 0.30)

    def tower(self, cx, y_base, w, h, emblem=False):
        col = lerp(self.pal[1], self.pal[2], 0.35)
        self.rect([cx - w // 2, y_base - h, cx + w // 2, y_base], col)
        r = self.rng
        layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for f in range(int(h / 34)):
            for i in range(int(w / 30)):
                if r.random() < 0.55:
                    wx = cx - w // 2 + 12 + i * 30
                    wy = y_base - h + 16 + f * 34
                    d.rectangle([wx, wy, wx + 18, wy + 20],
                                fill=rgb(lerp(self.pal[4], (200, 226, 255), 0.4)) + (int(r.integers(90, 210)),))
        self.img = Image.alpha_composite(self.img.convert('RGBA'), layer.filter(ImageFilter.GaussianBlur(1.4))).convert('RGB')
        if emblem:
            self.emblem(cx, y_base - h + 90, 82, glow=True)

    def emblem(self, cx, cy, r, glow=False):
        layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for k in range(8):
            a0 = math.radians(k * 45 - 90 + 22.5)
            a1 = math.radians((k + 1) * 45 - 90 + 22.5)
            pts = [(cx, cy), (cx + r * math.cos(a0), cy + r * math.sin(a0)), (cx + r * math.cos(a1), cy + r * math.sin(a1))]
            d.polygon(pts, fill=CRIMSON + (255,) if k % 2 == 0 else (238, 240, 244, 255))
        if glow:
            g = layer.filter(ImageFilter.GaussianBlur(r * 0.4))
            self.img = ImageChops.add(self.img, Image.eval(g.convert('RGB'), lambda v: int(v * 0.5)))
        self.img = Image.alpha_composite(self.img.convert('RGBA'), layer).convert('RGB')

    def corridor(self, vp=(0.5, 0.5), doors=True, cold=True):
        """One-point perspective corridor: the Arklay sublevel, an office hallway, a ship's spine."""
        vx, vy = int(W * vp[0]), int(H * vp[1])
        far = lerp(self.pal[1], self.pal[3], 0.5) if cold else lerp(self.pal[1], self.pal[2], 0.5)
        self.grad(self.pal[0], self.pal[0], 1.0)
        self.ellipse([vx - 420, vy - 300, vx + 420, vy + 300], far, blur=180, alpha=200)
        wall_l = lerp(self.pal[1], self.pal[2], 0.30)
        wall_r = lerp(self.pal[1], self.pal[2], 0.18)
        self.shape([(0, 0), (vx - 200, vy - 190), (vx - 200, vy + 190), (0, H)], wall_l)
        self.shape([(W, 0), (vx + 200, vy - 190), (vx + 200, vy + 190), (W, H)], wall_r)
        self.shape([(0, H), (vx - 200, vy + 190), (vx + 200, vy + 190), (W, H)], lerp(self.pal[1], self.pal[3], 0.22))
        self.shape([(0, 0), (vx - 200, vy - 190), (vx + 200, vy - 190), (W, 0)], lerp(self.pal[0], self.pal[1], 0.7))
        # ceiling light panels receding
        for k in range(1, 9):
            t = k / 9
            y = int(vy - 190 + (0 - (vy - 190)) * (1 - t) * 0.0 + 0)
            yy = int(vy - 190 - (vy - 190) * (1 - t))
            half = int((vx) * (1 - t) * 0.42) + 40
            self.rect([vx - half, yy - 12, vx + half, yy + 6], lerp(self.pal[3], self.pal[4], 0.55),
                      blur=10, alpha=int(70 + 150 * t))
        if doors:
            for k in range(1, 6):
                t = k / 6
                x_off = int((vx - 200) * (1 - t))
                top = int(vy - 190 * (1 - t) * 0.6 - 40 * t)
                bot = int(vy + 190 * (1 - t) * 0.6 + 60 * t)
                self.rect([x_off - 26, top, x_off + 4, bot], lerp(self.pal[1], self.pal[2], 0.55), alpha=210)
                self.rect([W - x_off - 4, top, W - x_off + 26, bot], lerp(self.pal[1], self.pal[2], 0.45), alpha=210)
        if self.has('emblem', 'umbrella', 'containment door', 'steel door'):
            self.emblem(vx, vy, 52, glow=False)

    def machine_room(self, machine=True, freezers=True):
        """A basement or plant room: back wall, floor, rows of cabinets, one big humming machine."""
        r = self.rng
        floor_y = int(H * 0.74)
        self.grad(lerp(self.pal[1], self.pal[2], 0.18), self.pal[0], 1.5, band=(0, floor_y))
        self.rect([0, floor_y, W, H], lerp(self.pal[0], self.pal[1], 0.42))
        # ceiling pipes
        for k in range(int(r.integers(3, 6))):
            y = int(H * (0.05 + k * 0.045))
            self.rect([0, y, W, y + int(r.integers(8, 20))], lerp(self.pal[0], self.pal[1], 0.60))
        # a strip light
        lx = int(W * r.uniform(0.3, 0.7))
        self.rect([lx - 150, int(H * 0.16), lx + 150, int(H * 0.18)], lerp(self.pal[3], self.pal[4], 0.65))
        self.glow(lx, int(H * 0.17), 340, lerp(self.pal[3], self.pal[4], 0.45), 0.42)
        if freezers:
            x = int(W * 0.04)
            while x < W * 0.62:
                w = int(r.integers(90, 140))
                h = int(r.integers(150, 210))
                self.rect([x, floor_y - h, x + w, floor_y], lerp(self.pal[1], self.pal[2], 0.30))
                self.rect([x + 8, floor_y - h + 10, x + w - 8, floor_y - h + 26],
                          lerp(self.pal[2], self.pal[3], 0.55), alpha=190)
                if r.random() < 0.5:
                    self.glow(x + w // 2, floor_y - h + 18, 60, lerp(self.pal[3], (150, 210, 255), 0.5), 0.30)
                x += w + int(r.integers(10, 40))
        if machine:
            mw, mh = int(W * 0.20), int(H * 0.46)
            mx = int(W * 0.78)
            self.rect([mx - mw // 2, floor_y - mh, mx + mw // 2, floor_y], lerp(self.pal[1], self.pal[2], 0.42))
            # vent slats
            for k in range(7):
                y = floor_y - mh + 40 + k * (mh - 90) / 7
                self.rect([mx - mw // 2 + 18, y, mx + mw // 2 - 18, y + 9], lerp(self.pal[0], self.pal[1], 0.5))
            # ducting to the ceiling
            self.rect([mx - 34, 0, mx + 34, floor_y - mh + 6], lerp(self.pal[1], self.pal[2], 0.34))
            # the little frosted window
            self.rect([mx - 40, floor_y - int(mh * 0.72), mx + 40, floor_y - int(mh * 0.50)],
                      lerp(self.pal[3], self.pal[4], 0.55), blur=5, alpha=210)
            self.glow(mx, floor_y - int(mh * 0.61), 120, lerp(self.pal[3], (190, 230, 255), 0.5), 0.40)
        self.fog(floor_y, 0.34)

    def room_interior(self, window=True, lamp=True):
        self.grad(lerp(self.pal[1], self.pal[2], 0.25), self.pal[0], 1.4)
        # back wall
        self.rect([0, 0, W, int(H * 0.78)], lerp(self.pal[0], self.pal[1], 0.75))
        self.rect([0, int(H * 0.78), W, H], lerp(self.pal[0], self.pal[1], 0.45))
        if window:
            wx0, wx1 = int(W * 0.52), int(W * 0.95)
            wy0, wy1 = int(H * 0.10), int(H * 0.62)
            self.rect([wx0, wy0, wx1, wy1], lerp(self.pal[2], self.pal[3], 0.45))
            r = self.rng
            for _ in range(60):
                x = r.integers(wx0, wx1)
                y = r.integers(wy0, wy1)
                self.rect([x, y, x + r.integers(3, 12), y + r.integers(3, 10)],
                          lerp(self.pal[4], (255, 200, 140), 0.5), alpha=int(r.integers(60, 190)))
            self.rect([wx0 - 6, wy0 - 6, wx0 + 6, wy1 + 6], self.pal[0])
            self.rect([wx0, wy0 - 8, wx1, wy0 + 6], self.pal[0])
            if self.has('rain'):
                self.rain_on_glass(wx0, wy0, wx1, wy1)
        if lamp:
            lx, ly = int(W * 0.22), int(H * 0.52)
            self.glow(lx, ly, 260, lerp(self.pal[3], (255, 214, 150), 0.6), 0.55)
            self.rect([lx - 120, ly + 60, lx + 240, ly + 78], lerp(self.pal[1], self.pal[2], 0.5))

    def rain_on_glass(self, x0, y0, x1, y1):
        layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        r = self.rng
        for _ in range(500):
            x = r.integers(x0, x1)
            y = r.integers(y0, y1)
            l = r.integers(6, 28)
            d.line([x, y, x - 2, y + l], fill=(230, 240, 255, int(r.integers(30, 90))), width=1)
        self.img = Image.alpha_composite(self.img.convert('RGBA'), layer).convert('RGB')

    def rain(self, density=900, wind=-3):
        layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        r = self.rng
        for _ in range(density):
            x = r.integers(-60, W)
            y = r.integers(0, H)
            l = r.integers(14, 46)
            d.line([x, y, x + wind * l / 10, y + l], fill=(220, 234, 255, int(r.integers(22, 80))), width=1)
        layer = layer.filter(ImageFilter.GaussianBlur(0.6))
        self.img = Image.alpha_composite(self.img.convert('RGBA'), layer).convert('RGB')

    def snow(self, density=700):
        layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        r = self.rng
        for _ in range(density):
            x, y = r.integers(0, W), r.integers(0, H)
            s = r.integers(1, 5)
            d.ellipse([x, y, x + s, y + s], fill=(235, 242, 252, int(r.integers(60, 190))))
        self.img = Image.alpha_composite(self.img.convert('RGBA'), layer.filter(ImageFilter.GaussianBlur(0.7))).convert('RGB')

    def embers(self, n=450):
        layer = Image.new('RGB', (W, H), (0, 0, 0))
        d = ImageDraw.Draw(layer)
        r = self.rng
        for _ in range(n):
            x, y = r.integers(0, W), r.integers(0, H)
            s = r.integers(1, 6)
            c = lerp((255, 140, 40), (255, 236, 170), r.random())
            d.ellipse([x, y, x + s, y + s], fill=rgb(c))
        self.img = ImageChops.add(self.img, layer.filter(ImageFilter.GaussianBlur(1.6)))

    def dust(self, n=260, color=None):
        layer = Image.new('RGB', (W, H), (0, 0, 0))
        d = ImageDraw.Draw(layer)
        r = self.rng
        c = color or lerp(self.pal[2], self.pal[3], 0.5)
        for _ in range(n):
            x, y = r.integers(0, W), r.integers(0, H)
            s = r.integers(1, 4)
            d.ellipse([x, y, x + s, y + s], fill=rgb(c))
        self.img = ImageChops.add(self.img, Image.eval(layer.filter(ImageFilter.GaussianBlur(2.2)), lambda v: int(v * 0.6)))

    def figure(self, cx, y_base, h, coat=True, facing=0, color=None, alpha=255):
        """A human silhouette. Long coat, shoulders, head; deliberately anonymous."""
        c = color or lerp(self.pal[0], self.pal[1], 0.35)
        hw = h * 0.16
        head_r = h * 0.072
        layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        neck = y_base - h + head_r * 2.1
        body = [(cx - hw * 0.72, neck + h * 0.02), (cx + hw * 0.72, neck + h * 0.02)]
        if coat:
            body += [(cx + hw * 1.02, y_base), (cx - hw * 1.02, y_base)]
        else:
            body += [(cx + hw * 0.78, y_base), (cx - hw * 0.78, y_base)]
        d.polygon(body, fill=rgb(c) + (alpha,))
        d.ellipse([cx - head_r, y_base - h, cx + head_r, y_base - h + head_r * 2], fill=rgb(c) + (alpha,))
        d.polygon([(cx - hw * 0.72, neck + h * 0.03), (cx - hw * 1.18, neck + h * 0.34),
                   (cx - hw * 0.86, neck + h * 0.36), (cx - hw * 0.5, neck + h * 0.08)], fill=rgb(c) + (alpha,))
        d.polygon([(cx + hw * 0.72, neck + h * 0.03), (cx + hw * 1.18, neck + h * 0.34),
                   (cx + hw * 0.86, neck + h * 0.36), (cx + hw * 0.5, neck + h * 0.08)], fill=rgb(c) + (alpha,))
        self.img = Image.alpha_composite(self.img.convert('RGBA'), layer).convert('RGB')

    def rim_light(self, cx, y_base, h, color=None):
        c = color or lerp(self.pal[3], self.pal[4], 0.6)
        layer = Image.new('RGB', (W, H), (0, 0, 0))
        d = ImageDraw.Draw(layer)
        hw = h * 0.16
        d.line([(cx + hw * 0.74, y_base - h * 0.78), (cx + hw * 1.04, y_base)], fill=rgb(c), width=4)
        d.arc([cx - h * 0.072, y_base - h, cx + h * 0.072, y_base - h + h * 0.144], -70, 30, fill=rgb(c), width=4)
        self.img = ImageChops.add(self.img, layer.filter(ImageFilter.GaussianBlur(4)))

    def crowd(self, y_base, rows=7, color=None):
        r = self.rng
        for row in range(rows):
            t = row / max(1, rows - 1)
            h = int(H * (0.10 + 0.16 * t))
            y = int(y_base + t * H * 0.18)
            c = color or lerp(self.pal[0], self.pal[1], 0.15 + 0.3 * (1 - t))
            x = -40
            while x < W + 40:
                self.figure(x, y, h, coat=False, color=c, alpha=255)
                x += int(h * 0.30 + r.integers(0, int(h * 0.22)))

    def ship(self, cx, cy, length, color=None, alien=False):
        r = self.rng
        c = color or lerp(self.pal[0], self.pal[1], 0.5)
        hgt = length * (0.26 if not alien else 0.38)
        if alien:
            pts = [(cx - length / 2, cy), (cx - length * 0.2, cy - hgt), (cx + length * 0.32, cy - hgt * 0.62),
                   (cx + length / 2, cy + hgt * 0.14), (cx + length * 0.1, cy + hgt), (cx - length * 0.3, cy + hgt * 0.5)]
        else:
            pts = [(cx - length / 2, cy - hgt * 0.42), (cx + length * 0.30, cy - hgt * 0.30),
                   (cx + length / 2, cy - hgt * 0.04), (cx + length * 0.34, cy + hgt * 0.26),
                   (cx - length / 2, cy + hgt * 0.46)]
        self.shape(pts, c)
        edge = Image.new('RGB', (W, H), (0, 0, 0))
        de = ImageDraw.Draw(edge)
        rim = rgb(lerp(self.pal[3], self.pal[4], 0.45))
        de.line(pts + [pts[0]], fill=rim, width=2)
        # hull panel lines
        for k in range(int(length // 70)):
            t = (k + 1) / (length // 70 + 1)
            x = cx - length / 2 + length * t
            de.line([(x, cy - hgt * 0.18), (x, cy + hgt * 0.16)],
                    fill=rgb(lerp(self.pal[1], self.pal[3], 0.35)), width=1)
        # running lights
        for k in range(max(2, int(length // 130))):
            t = (k + 0.5) / max(2, int(length // 130))
            x = cx - length / 2 + length * t
            de.ellipse([x - 2, cy - hgt * 0.20, x + 2, cy - hgt * 0.20 + 4],
                       fill=rgb(lerp(CRIMSON, (255, 220, 220), 0.25)))
        self.img = ImageChops.add(self.img, Image.eval(edge.filter(ImageFilter.GaussianBlur(1.4)), lambda v: int(v * 0.9)))
        # engine glow at the stern
        ex = cx - length / 2
        gc = lerp(self.pal[3], (150, 200, 255), 0.55) if not alien else lerp(self.pal[3], (170, 255, 210), 0.5)
        self.glow(int(ex), int(cy + hgt * 0.06), int(max(18, length * 0.10)), gc, 0.55)

    def planet(self, cx, cy, r, ring=False):
        base = lerp(self.pal[2], self.pal[3], 0.4)
        self.ellipse([cx - r, cy - r, cx + r, cy + r], base)
        self.ellipse([cx - r * 0.96, cy - r * 0.96, cx + r * 0.6, cy + r * 0.6], lerp(base, self.pal[4], 0.3), blur=40, alpha=110)
        self.ellipse([cx - r * 1.06, cy - r * 1.06, cx + r * 1.06, cy + r * 1.06],
                     lerp(self.pal[3], self.pal[4], 0.5), blur=48, alpha=60)
        # night side
        self.ellipse([cx - r * 0.5, cy - r, cx + r * 1.4, cy + r], self.pal[0], blur=30, alpha=190)

    def cryo_tank(self, cx, cy, w, h):
        """A cryogenic cradle: steel body, a frosted window, and something enormous half-seen inside."""
        r = self.rng
        body = lerp(self.pal[1], self.pal[2], 0.42)
        self.rect([cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2], body)
        # ribs and plating
        for k in range(6):
            y = cy - h // 2 + 30 + k * (h - 70) / 6
            self.rect([cx - w // 2 + 6, y, cx + w // 2 - 6, y + 7], lerp(self.pal[0], self.pal[1], 0.55), alpha=170)
        # cabling to the ceiling
        for k in range(3):
            x = cx - w * 0.22 + k * w * 0.22
            self.rect([x - 7, 0, x + 7, cy - h // 2 + 10], lerp(self.pal[1], self.pal[2], 0.30))

        # the window, and the shape behind it
        gx0, gy0 = cx - int(w * 0.31), cy - int(h * 0.34)
        gx1, gy1 = cx + int(w * 0.31), cy + int(h * 0.20)
        inner = Image.new('RGB', (W, H), (0, 0, 0))
        di = ImageDraw.Draw(inner)
        di.rectangle([gx0, gy0, gx1, gy1], fill=rgb(lerp(self.pal[2], self.pal[4], 0.30)))
        self.img.paste(inner.crop((gx0, gy0, gx1, gy1)), (gx0, gy0))

        # occupant: deliberately indistinct, larger than the window, blurred into the mist
        occ = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        do = ImageDraw.Draw(occ)
        fh = int((gy1 - gy0) * 1.35)
        fw = int(fh * 0.34)
        top = gy1 - fh + int(fh * 0.12)
        dark = rgb(lerp(self.pal[0], self.pal[1], 0.30))
        do.rounded_rectangle([cx - fw // 2, top + int(fh * 0.20), cx + fw // 2, gy1 + 40],
                             radius=int(fw * 0.35), fill=dark + (235,))
        do.ellipse([cx - int(fw * 0.30), top, cx + int(fw * 0.30), top + int(fw * 0.62)], fill=dark + (235,))
        # shoulders wider than a person's
        do.polygon([(cx - fw * 0.78, top + fh * 0.36), (cx + fw * 0.78, top + fh * 0.36),
                    (cx + fw * 0.52, top + fh * 0.58), (cx - fw * 0.52, top + fh * 0.58)], fill=dark + (225,))
        # arms hanging
        do.rounded_rectangle([cx - fw * 0.86, top + fh * 0.36, cx - fw * 0.58, gy1 + 10],
                             radius=int(fw * 0.2), fill=dark + (215,))
        do.rounded_rectangle([cx + fw * 0.58, top + fh * 0.36, cx + fw * 0.86, gy1 + 10],
                             radius=int(fw * 0.2), fill=dark + (215,))
        occ = occ.filter(ImageFilter.GaussianBlur(6))
        mask = Image.new('L', (W, H), 0)
        ImageDraw.Draw(mask).rectangle([gx0 + 3, gy0 + 3, gx1 - 3, gy1 - 3], fill=255)
        occ.putalpha(ImageChops.multiply(occ.split()[3], mask))
        self.img = Image.alpha_composite(self.img.convert('RGBA'), occ).convert('RGB')

        # vapour inside the glass
        mist = Image.new('RGB', (W, H), (0, 0, 0))
        dm = ImageDraw.Draw(mist)
        for _ in range(14):
            mx = r.integers(gx0, gx1)
            my = r.integers(gy0, gy1)
            rr = r.integers(40, 150)
            dm.ellipse([mx - rr, my - rr * 0.5, mx + rr, my + rr * 0.5],
                       fill=rgb(lerp(self.pal[3], self.pal[4], 0.7)))
        mist = mist.filter(ImageFilter.GaussianBlur(46))
        mist = ImageChops.multiply(mist, Image.merge('RGB', [mask] * 3))
        self.img = ImageChops.add(self.img, Image.eval(mist, lambda v: int(v * 0.26)))

        # frost creeping in from the edges of the glass
        frost = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        df = ImageDraw.Draw(frost)
        for _ in range(90):
            fx = r.integers(gx0, gx1)
            fy = r.choice([r.integers(gy0, gy0 + 60), r.integers(gy1 - 60, gy1)])
            rr = r.integers(6, 34)
            df.ellipse([fx - rr, fy - rr, fx + rr, fy + rr],
                       fill=rgb(lerp(self.pal[3], self.pal[4], 0.85)) + (int(r.integers(20, 70)),))
        frost = frost.filter(ImageFilter.GaussianBlur(7))
        frost.putalpha(ImageChops.multiply(frost.split()[3], mask))
        self.img = Image.alpha_composite(self.img.convert('RGBA'), frost).convert('RGB')

        # glass frame and rim light
        edge = Image.new('RGB', (W, H), (0, 0, 0))
        ImageDraw.Draw(edge).rectangle([gx0, gy0, gx1, gy1],
                                       outline=rgb(lerp(self.pal[3], self.pal[4], 0.55)), width=4)
        self.img = ImageChops.add(self.img, edge.filter(ImageFilter.GaussianBlur(2)))
        self.glow(cx, (gy0 + gy1) // 2, int(w * 0.62), lerp(self.pal[3], (170, 215, 255), 0.45), 0.34)
        # cold spill onto the floor
        self.glow(cx, cy + h // 2, int(w * 0.7), lerp(self.pal[2], self.pal[3], 0.5), 0.22)

    def hologram(self, cx, cy, w, h):
        self.rect([cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2], (10, 12, 18), alpha=190)
        edge = Image.new('RGB', (W, H), (0, 0, 0))
        d = ImageDraw.Draw(edge)
        d.rectangle([cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2], outline=CRIMSON, width=3)
        r = self.rng
        for k in range(int(h / 34)):
            y = cy - h // 2 + 26 + k * 34
            d.line([cx - w // 2 + 26, y, cx - w // 2 + 26 + r.integers(60, w - 80), y],
                   fill=rgb(lerp((200, 210, 230), CRIMSON, 0.2)), width=3)
        self.img = ImageChops.add(self.img, edge.filter(ImageFilter.GaussianBlur(1.4)))
        g = edge.filter(ImageFilter.GaussianBlur(26))
        self.img = ImageChops.add(self.img, Image.eval(g, lambda v: int(v * 0.55)))

    def screens(self, n=6):
        r = self.rng
        for _ in range(n):
            w = r.integers(150, 380)
            h = int(w * 0.6)
            x = r.integers(40, W - w - 40)
            y = r.integers(60, int(H * 0.7))
            self.rect([x, y, x + w, y + h], lerp(self.pal[1], self.pal[2], 0.55), alpha=230)
            self.rect([x + 8, y + 8, x + w - 8, y + h - 8], lerp(self.pal[2], self.pal[3], 0.6), blur=6, alpha=190)
            self.glow(x + w // 2, y + h // 2, int(w * 0.5), lerp(self.pal[3], self.pal[4], 0.3), 0.22)

    def texture(self, amount=0.05, scale=3):
        """Break up flat fills with low-frequency mottling so nothing reads as vector art."""
        small = Image.fromarray(
            (self.rng.normal(0.5, 0.16, (H // (scale * 8), W // (scale * 8), 3)).clip(0, 1) * 255).astype(np.uint8))
        noise = small.resize((W, H), Image.BICUBIC).filter(ImageFilter.GaussianBlur(scale * 2))
        a = np.array(self.img).astype(np.float32)
        n = np.array(noise).astype(np.float32) - 127.0
        a = np.clip(a + n * amount, 0, 255)
        self.img = Image.fromarray(a.astype(np.uint8))

    def foreground_frame(self, strength=1.0):
        """Near-black out-of-focus shapes at the edges: instant depth."""
        r = self.rng
        layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        dark = lerp(self.pal[0], (0, 0, 0), 0.55)
        side = r.integers(0, 3)
        if side in (0, 2):
            w = int(W * r.uniform(0.08, 0.20))
            d.polygon([(0, 0), (w, 0), (int(w * r.uniform(0.7, 1.1)), H), (0, H)], fill=rgb(dark) + (255,))
        if side in (1, 2):
            w = int(W * r.uniform(0.08, 0.18))
            d.polygon([(W, 0), (W - w, 0), (W - int(w * r.uniform(0.7, 1.1)), H), (W, H)], fill=rgb(dark) + (255,))
        if r.random() < 0.55:
            h = int(H * r.uniform(0.05, 0.13))
            d.rectangle([0, H - h, W, H], fill=rgb(dark) + (255,))
        layer = layer.filter(ImageFilter.GaussianBlur(int(18 * strength)))
        self.img = Image.alpha_composite(self.img.convert('RGBA'), layer).convert('RGB')

    def wet_ground(self, y, height=None):
        """Reflect the scene below a horizon line and smear it: wet asphalt, sea, polished floor."""
        height = height or (H - y)
        if height < 8 or y < 8:
            return
        top = self.img.crop((0, max(0, y - height), W, y)).transpose(Image.FLIP_TOP_BOTTOM)
        top = top.filter(ImageFilter.GaussianBlur(7))
        a = np.array(top).astype(np.float32)
        fade = np.linspace(0.52, 0.06, a.shape[0])[:, None, None]
        a = a * fade
        # horizontal smear
        a = (a + np.roll(a, 5, axis=1) + np.roll(a, -5, axis=1)) / 3.0
        self.img.paste(Image.fromarray(a.astype(np.uint8)), (0, y))

    # ------------------------------------------------------------- grading
    def god_rays(self, cx, cy, n=9, strength=0.35):
        layer = Image.new('RGB', (W, H), (0, 0, 0))
        d = ImageDraw.Draw(layer)
        r = self.rng
        for _ in range(n):
            a = r.uniform(0, math.pi * 2)
            w = r.uniform(0.02, 0.09)
            L = 2400
            p1 = (cx + math.cos(a - w) * L, cy + math.sin(a - w) * L)
            p2 = (cx + math.cos(a + w) * L, cy + math.sin(a + w) * L)
            d.polygon([(cx, cy), p1, p2], fill=rgb(lerp(self.pal[3], self.pal[4], 0.55)))
        layer = layer.filter(ImageFilter.GaussianBlur(120))
        self.img = ImageChops.add(self.img, Image.eval(layer, lambda v: int(v * strength)))

    def fog(self, y, strength=0.5):
        layer = Image.new('RGB', (W, H), (0, 0, 0))
        d = ImageDraw.Draw(layer)
        r = self.rng
        for _ in range(16):
            cx, cy = r.integers(-200, W + 200), int(y + r.integers(-140, 160))
            rr = r.integers(280, 760)
            d.ellipse([cx - rr, cy - rr * 0.22, cx + rr, cy + rr * 0.22], fill=rgb(lerp(self.pal[2], self.pal[3], 0.55)))
        self.img = ImageChops.add(self.img, Image.eval(layer.filter(ImageFilter.GaussianBlur(140)), lambda v: int(v * strength)))

    def grade(self):
        a = np.array(self.img).astype(np.float32) / 255.0
        # filmic contrast
        a = np.clip((a - 0.5) * 1.12 + 0.5, 0, 1)
        lum = a.mean(axis=2, keepdims=True)
        a = lum + (a - lum) * 0.92
        # vignette
        yy, xx = np.mgrid[0:H, 0:W]
        d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        a *= (1 - 0.42 * np.clip(d - 0.35, 0, 1) ** 1.5)[..., None]
        # Exposure floor. Night, space and basement palettes can stack a dark gradient under
        # dark subject matter and land on a frame that reads as flat black on screen -- and a
        # shot holds for the better part of a minute, so there is nowhere for the eye to go.
        # A gamma lift is used rather than a gain because it pins both endpoints: the blacks
        # stay black and the highlights stay put, while the tones in between come up.
        # measured as ITU-R 601 luma, not as a flat channel average: these palettes run
        # blue, and blue carries barely a tenth of the perceived brightness, so a mean
        # would read a cold frame as far lighter than the eye finds it
        luma = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
        bright = float(np.percentile(luma, 99))
        if 1e-3 < bright < FLOOR_P99:
            gamma = max(0.45, math.log(LIFT_TO_P99) / math.log(bright))
            a = np.clip(a, 0, 1) ** gamma
        # grain
        g = self.rng.normal(0, 0.012, a.shape[:2])[..., None]
        a = np.clip(a + g, 0, 1)
        # slight chromatic lift in shadows
        a[..., 2] += 0.016 * (1 - lum[..., 0])
        a = np.clip(a, 0, 1)
        self.img = Image.fromarray((a * 255).astype(np.uint8))

    # ------------------------------------------------------------- compose
    def compose(self):
        p = self.p
        r = self.rng
        # ---- space / fleets
        if self.has('space', 'orbit', 'starfield', 'interstellar', 'fleet', 'asteroid', 'void', 'nebula',
                    'alien vessel', 'alien fleet', 'warship', 'ship ', 'ships', 'megastructure', 'dock'):
            self.sky()
            if self.has('earth', 'planet', 'mars', 'world', 'orbit', 'moon') or r.random() < 0.55:
                self.planet(int(W * r.uniform(0.08, 0.42)), int(H * r.uniform(0.74, 1.18)), int(H * r.uniform(0.55, 1.0)))
            alien = self.has('alien', 'cathedral-like', 'veth', 'hive', 'organic', 'choir', 'myriad', 'tollhouse')
            if self.has('fleet', 'hundreds', 'formation', 'swarm', 'dozens'):
                for _ in range(int(r.integers(9, 18))):
                    self.ship(int(r.integers(0, W)), int(r.integers(int(H * 0.2), int(H * 0.85))),
                              int(r.integers(90, 320)), alien=alien)
            self.ship(int(W * r.choice([0.36, 0.62])), int(H * r.uniform(0.36, 0.60)),
                      int(W * r.uniform(0.46, 0.78)), alien=alien)
            if self.has('flash', 'explosion', 'engaging', 'battle', 'firing', 'laser', 'intercept'):
                for _ in range(int(r.integers(3, 7))):
                    self.glow(int(r.integers(0, W)), int(r.integers(0, H)), int(r.integers(40, 150)),
                              (255, 230, 190), r.uniform(0.5, 1.0))
            self.dust(120)
            self.texture(0.025, scale=5)
            self.grade()
            return self.img

        # ---- cryogenic cradles (before corridors: these are often "at the end of a corridor")
        if self.has('cryogenic', 'cryo', 'capsule', 'cradle', 'frosted glass', 'viewing window', 'frosted'):
            self.grad(lerp(self.pal[1], self.pal[2], 0.28), self.pal[0], 1.5)
            self.rect([0, int(H * 0.80), W, H], lerp(self.pal[0], self.pal[1], 0.40))
            self.cryo_tank(int(W * r.uniform(0.44, 0.60)), int(H * 0.50), int(W * 0.30), int(H * 0.74))
            if self.has('figure', 'standing', 'hand', 'watching', 'sitting', 'palm'):
                self.figure(int(W * r.uniform(0.18, 0.30)), int(H * 0.94), int(H * 0.44))
                self.rim_light(int(W * r.uniform(0.18, 0.30)), int(H * 0.94), int(H * 0.44))
            self.fog(int(H * 0.82), 0.40)
            self.texture(0.05)
            self.foreground_frame(0.7)
            self.grade()
            return self.img

        # ---- basements, plant rooms, freezer halls
        if self.has('basement', 'freezer', 'freezers', 'hvac', 'plant room', 'boiler', 'loading dock',
                    'machine in the corner', 'server', 'data centre', 'data center'):
            self.machine_room()
            if self.has('figure', 'man', 'men', 'standing', 'walking', 'agents', 'guard', 'wesker', 'flashlight'):
                for k in range(int(r.integers(1, 3))):
                    self.figure(int(W * r.uniform(0.2, 0.6)), int(H * r.uniform(0.76, 0.84)), int(H * 0.30))
            self.dust(90, color=lerp(self.pal[2], self.pal[3], 0.4))
            self.texture(0.055)
            self.foreground_frame(0.8)
            self.grade()
            return self.img

        # ---- corridors and labs
        if self.has('corridor', 'sublevel', 'white corridor', 'hallway', 'tunnel'):
            self.corridor(vp=(r.uniform(0.44, 0.56), r.uniform(0.46, 0.54)))
            if self.has('figure', 'man', 'woman', 'walking', 'standing', 'agent', 'wesker', 'alone'):
                self.figure(int(W * r.uniform(0.42, 0.58)), int(H * 0.76), int(H * 0.36))
                self.rim_light(int(W * r.uniform(0.42, 0.58)), int(H * 0.76), int(H * 0.36))
            self.fog(int(H * 0.62), 0.30)
            self.wet_ground(int(H * 0.70), int(H * 0.26))
            self.texture(0.045)
            self.foreground_frame(0.8)
            self.grade()
            return self.img

        if self.has('laboratory', 'lab ', 'microscope', 'bench', 'clean room', 'sequencer'):
            self.room_interior(window=self.has('window', 'sea', 'earth'), lamp=True)
            self.screens(int(r.integers(3, 7)))
            if self.has('figure', 'scientist', 'woman', 'man', 'evelyn', 'wesker', 'standing', 'working'):
                self.figure(int(W * r.uniform(0.28, 0.62)), int(H * 0.94), int(H * 0.52))
            self.dust(160)
            self.texture(0.05)
            self.foreground_frame(0.6)
            self.grade()
            return self.img

        # ---- holographic system panels
        if self.has('holographic', 'hologram', 'interface panel', 'glowing crimson', 'system panel', 'shop'):
            self.grad(lerp(self.pal[1], self.pal[2], 0.2), self.pal[0], 1.7)
            for k in range(int(r.integers(2, 5))):
                self.hologram(int(W * r.uniform(0.3, 0.7)), int(H * r.uniform(0.3, 0.7)),
                              int(W * r.uniform(0.22, 0.42)), int(H * r.uniform(0.18, 0.4)))
            if self.has('man', 'figure', 'wesker', 'standing', 'silhouette'):
                self.figure(int(W * 0.24), int(H * 0.95), int(H * 0.56))
                self.rim_light(int(W * 0.24), int(H * 0.95), int(H * 0.56), CRIMSON)
            self.dust(200)
            self.grade()
            return self.img

        # ---- crowds, stadiums, squares
        if self.has('crowd', 'stadium', 'square filled', 'thousands of people', 'auditorium', 'rally',
                    'assembly', 'hundreds of thousands', 'people in the stands'):
            self.sky()
            self.skyline(int(H * 0.52), depth=0.6, tall=True)
            self.fog(int(H * 0.54), 0.35)
            if self.has('emblem', 'umbrella', 'drone'):
                self.emblem(W // 2, int(H * 0.26), 150, glow=True)
            self.crowd(int(H * 0.60), rows=8)
            if self.has('stage', 'balcony', 'podium', 'speaking'):
                self.rect([0, int(H * 0.58), W, int(H * 0.63)], lerp(self.pal[0], self.pal[1], 0.4))
                self.figure(W // 2, int(H * 0.59), int(H * 0.18))
                self.glow(W // 2, int(H * 0.52), 260, lerp(self.pal[3], self.pal[4], 0.5), 0.5)
            self.dust(200)
            self.texture(0.04)
            self.foreground_frame(0.7)
            self.grade()
            return self.img

        # ---- the old brick building / industrial park
        if self.has('brick', 'industrial park', 'four-storey', 'old building', 'division seven'):
            self.sky()
            self.skyline(int(H * 0.70), depth=0.8, tall=False)
            gy = int(H * 0.86)
            self.brick_building(int(W * 0.5), gy, int(W * 0.34), int(H * 0.44))
            if self.has('sign', 'emblem', 'umbrella'):
                self.emblem(int(W * 0.5), int(H * 0.44), 44)
            self.fog(int(H * 0.80), 0.26)
            self.rect([0, gy, W, H], lerp(self.pal[0], self.pal[1], 0.22))
            self.wet_ground(gy, int(H * 0.34))
            for k in range(int(r.integers(2, 5))):
                x = int(W * r.uniform(0.05, 0.95))
                self.glow(x, gy - 40, 90, lerp(self.pal[3], (255, 196, 120), 0.6), 0.35)
                self.rect([x - 3, gy - 150, x + 3, gy], lerp(self.pal[0], self.pal[1], 0.5))
            if self.has('figure', 'man', 'standing', 'guard', 'walking'):
                self.figure(int(W * r.uniform(0.2, 0.8)), int(H * 0.95), int(H * 0.30))
            if self.has('rain'):
                self.rain(1100)
            self.texture(0.05)
            self.foreground_frame(1.0)
            self.grade()
            return self.img

        # ---- city / skyline / tower
        if self.has('city', 'skyline', 'tower', 'street', 'downtown', 'manhattan', 'rooftop', 'basel',
                    'geneva', 'beijing', 'london', 'ocean city', 'arcology', 'settlement', 'colony'):
            self.sky()
            self.skyline(int(H * 0.62), depth=0.85, tall=True)
            self.skyline(int(H * 0.72), depth=0.45, tall=True)
            if self.has('tower', 'umbrella', 'emblem', 'headquarters'):
                self.tower(int(W * r.uniform(0.34, 0.66)), int(H * 0.82), int(W * 0.14), int(H * 0.60),
                           emblem=self.has('emblem', 'umbrella'))
            self.skyline(int(H * 0.84), depth=0.12, tall=False)
            self.horizon_haze(int(H * 0.62))
            if self.has('rain'):
                self.rain(900)
            if self.has('snow', 'winter'):
                self.snow()
            gy = int(H * 0.86)
            self.rect([0, gy, W, H], lerp(self.pal[0], self.pal[1], 0.2))
            self.wet_ground(gy, int(H * 0.30))
            if self.has('figure', 'man', 'woman', 'walking', 'standing', 'alone', 'wesker'):
                self.figure(int(W * r.uniform(0.18, 0.82)), int(H * 0.99), int(H * 0.42))
            self.dust(140)
            self.texture(0.05)
            self.foreground_frame(1.0)
            self.grade()
            return self.img

        # ---- desert
        if self.has('desert', 'mojave', 'dunes', 'compound'):
            self.sky()
            self.shape([(0, int(H * 0.70)), (W, int(H * 0.66)), (W, H), (0, H)], lerp(self.pal[1], self.pal[2], 0.45))
            self.rect([int(W * 0.3), int(H * 0.60), int(W * 0.72), int(H * 0.70)], lerp(self.pal[0], self.pal[1], 0.5))
            for k in range(4):
                x = int(W * (0.34 + k * 0.10))
                self.rect([x, int(H * 0.56), x + 40, int(H * 0.60)], lerp(self.pal[0], self.pal[1], 0.4))
            self.fog(int(H * 0.70), 0.45)
            if self.has('figure', 'men', 'soldiers', 'agents', 'giant', 'convoy'):
                for k in range(int(r.integers(3, 9))):
                    self.figure(int(W * r.uniform(0.15, 0.85)), int(H * r.uniform(0.80, 0.95)), int(H * 0.16))
            self.dust(320)
            self.grade()
            return self.img

        # ---- volcano / fire
        if self.has('volcano', 'lava', 'fire', 'erupting', 'burning'):
            self.grad(lerp(self.pal[2], self.pal[3], 0.4), self.pal[0], 1.4)
            self.glow(int(W * 0.5), int(H * 0.82), 700, (255, 120, 30), 0.65)
            self.shape([(0, int(H * 0.62)), (int(W * 0.34), int(H * 0.40)), (int(W * 0.62), int(H * 0.46)),
                        (W, int(H * 0.58)), (W, H), (0, H)], self.pal[0])
            self.rect([0, int(H * 0.86), W, H], (150, 46, 12), blur=60, alpha=200)
            self.embers()
            if self.has('figure', 'man', 'silhouette'):
                self.figure(int(W * 0.5), int(H * 0.92), int(H * 0.34), color=(8, 4, 4))
                self.rim_light(int(W * 0.5), int(H * 0.92), int(H * 0.34), (255, 160, 60))
            self.grade()
            return self.img

        # ---- bunker / 1945
        if self.has('bunker', 'underground corridor', '1945', 'air raid'):
            self.corridor(vp=(0.5, 0.5), doors=False, cold=False)
            self.glow(W // 2, int(H * 0.30), 200, (240, 214, 150), 0.5)
            self.dust(420, color=(190, 170, 130))
            self.grade()
            return self.img

        # ---- interiors: office, boardroom, courtroom, diner, church, home
        if self.has('office', 'boardroom', 'desk', 'conference', 'courtroom', 'diner', 'restaurant',
                    'kitchen', 'apartment', 'church', 'room', 'chamber', 'hearing', 'table', 'booth', 'newsroom'):
            self.room_interior(window=not self.has('windowless', 'no windows'),
                               lamp=True)
            if self.has('screen', 'monitor', 'newsroom', 'operations', 'display', 'television'):
                self.screens(int(r.integers(2, 6)))
            n = 1
            if self.has('board', 'delegates', 'senators', 'committee', 'gathered', 'meeting', 'staff', 'crowd'):
                n = int(r.integers(4, 9))
            for k in range(n):
                self.figure(int(W * (0.18 + 0.64 * (k + 0.5) / n + r.uniform(-0.03, 0.03))),
                            int(H * r.uniform(0.93, 1.0)), int(H * r.uniform(0.40, 0.52)))
            self.dust(140)
            self.texture(0.05)
            self.foreground_frame(0.7)
            self.grade()
            return self.img

        # ---- close-ups: eyes, hands, objects
        if self.has('close-up', 'macro', 'extreme close'):
            self.grad(lerp(self.pal[1], self.pal[2], 0.3), self.pal[0], 1.8)
            self.glow(int(W * r.uniform(0.35, 0.65)), int(H * r.uniform(0.35, 0.6)), 460,
                      lerp(self.pal[2], self.pal[3], 0.6), 0.5)
            if self.has('eye', 'eyes', 'pupil'):
                cx, cy = W // 2, H // 2
                self.ellipse([cx - 420, cy - 210, cx + 420, cy + 210], lerp(self.pal[1], self.pal[2], 0.5), blur=10)
                self.ellipse([cx - 175, cy - 175, cx + 175, cy + 175], (150, 30, 24) if self.has('red', 'glow', 'amber')
                             else lerp(self.pal[2], self.pal[3], 0.55))
                self.ellipse([cx - 30, cy - 150, cx + 30, cy + 150], (8, 6, 8))
                if self.has('red', 'glow', 'amber'):
                    self.glow(cx, cy, 240, (190, 40, 30), 0.65)
                self.ellipse([cx + 55, cy - 95, cx + 110, cy - 40], (250, 250, 255), blur=8, alpha=200)
            else:
                self.rect([int(W * 0.18), int(H * 0.40), int(W * 0.82), int(H * 0.72)],
                          lerp(self.pal[1], self.pal[2], 0.55), blur=4)
                self.glow(int(W * 0.5), int(H * 0.56), 220, lerp(self.pal[3], self.pal[4], 0.4), 0.35)
            self.dust(120)
            self.grade()
            return self.img

        # ---- default: atmospheric establishing shot
        self.sky()
        self.skyline(int(H * 0.60), depth=0.85, tall=True)
        self.horizon_haze(int(H * 0.62))
        self.skyline(int(H * 0.74), depth=0.45, tall=False)
        self.fog(int(H * 0.70), 0.38)
        self.skyline(int(H * 0.88), depth=0.06, tall=False)
        gy = int(H * 0.88)
        self.rect([0, gy, W, H], lerp(self.pal[0], self.pal[1], 0.18))
        self.wet_ground(gy, int(H * 0.26))
        if self.has('rain'):
            self.rain(900)
        if self.has('figure', 'man', 'woman', 'standing', 'walking', 'alone'):
            self.figure(int(W * r.choice([0.3, 0.68])), int(H * 0.97), int(H * 0.40))
        self.dust(140)
        self.texture(0.05)
        self.foreground_frame(1.0)
        self.grade()
        return self.img


def strip_tags(prompt):
    """Expand {NAME} character tags into their look-sheet description."""
    import json
    path = os.path.join(ROOT, 'story', 'visuals.json')
    looks = json.load(open(path)) if os.path.exists(path) else {}
    def sub(m):
        return looks.get(m.group(1), m.group(1).lower())
    return re.sub(r'\{([A-Z_]+)\}', sub, prompt)


def render(prompt, shot_id):
    seed = int(hashlib.sha1(shot_id.encode()).hexdigest()[:8], 16)
    return Scene(strip_tags(prompt), seed).compose()


def render_chapter(n, force=False):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from parse import parse
    from audio import shot_id
    ch = parse(os.path.join(ROOT, 'script', f'ch{n:02d}.txt'))
    os.makedirs(OUT, exist_ok=True)
    made = 0
    for e in ch.events:
        if e.kind != 'img':
            continue
        sid = shot_id(n, e.text)
        path = os.path.join(OUT, f'{sid}.png')
        if os.path.exists(path) and not force:
            continue
        render(e.text, sid).save(path, optimize=True)
        made += 1
        # second variant for long shots, from a different seed
        p2 = os.path.join(OUT, f'{sid}_v1.png')
        if not os.path.exists(p2) or force:
            Scene(strip_tags(e.text), int(hashlib.sha1((sid + 'v1').encode()).hexdigest()[:8], 16)).compose().save(p2, optimize=True)
    return made


if __name__ == '__main__':
    if '--demo' in sys.argv:
        os.makedirs('/tmp/claude-0/art', exist_ok=True)
        demos = [
            'exterior of a tired four-storey brick research building in an industrial park at night in heavy rain, a faded blue sign, sodium streetlights, empty parking lot, cinematic',
            'a vast pristine underground laboratory corridor with white walls, glass-walled containment chambers, cool blue light, cinematic wide shot',
            'an enormous football stadium at dusk packed with sixty thousand people, a giant stage, cinematic wide shot',
            'a dark office at night lit only by rows of floating crimson and white holographic shop panels, a man silhouetted before them, cinematic',
            'an erupting volcano interior, rivers of lava, apocalyptic orange light, smoke, cinematic wide shot',
            'a vast dark wedge-shaped human warship coasting through deep space, stars, cinematic',
            'extreme close-up of a man\'s eye in darkness, the pupil narrowing into a vertical slit, glowing faint amber-red, cinematic',
            'a cryogenic cradle in a cold room, a towering grey figure asleep in pale mist behind frosted glass, cinematic',
            'a gleaming biotech tower at night with a red and white umbrella emblem, city skyline, cinematic',
            'a concrete bunker corridor deep underground, 1945, a single bare bulb, dust falling, cinematic',
        ]
        for i, d in enumerate(demos):
            render(d, f'demo{i}').save(f'/tmp/claude-0/art/demo{i}.png')
            print(f'demo{i}')
    elif '--chapter' in sys.argv:
        n = int(sys.argv[sys.argv.index('--chapter') + 1])
        print(n, render_chapter(n, force='--force' in sys.argv))
