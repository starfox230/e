"""Static graphics: System panels, location captions, title/chapter/movement cards, the Umbrella mark."""
import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H = 1920, 1080
FONT_DIR = '/usr/share/fonts/truetype'


def font(kind, size):
    paths = {
        'sans': ['dejavu/DejaVuSans.ttf'],
        'sans_b': ['dejavu/DejaVuSans-Bold.ttf'],
        'cond': ['dejavu/DejaVuSansCondensed.ttf'],
        'cond_b': ['dejavu/DejaVuSansCondensed-Bold.ttf'],
        'serif': ['dejavu/DejaVuSerif.ttf', 'liberation/LiberationSerif-Regular.ttf'],
        'serif_b': ['dejavu/DejaVuSerif-Bold.ttf', 'liberation/LiberationSerif-Bold.ttf'],
        'mono': ['dejavu/DejaVuSansMono.ttf'],
        'mono_b': ['dejavu/DejaVuSansMono-Bold.ttf'],
    }[kind]
    for p in paths:
        full = os.path.join(FONT_DIR, p)
        if os.path.exists(full):
            return ImageFont.truetype(full, size)
    return ImageFont.load_default()


CRIMSON = (196, 22, 36)
WHITE = (240, 242, 245)
GLASS = (14, 16, 22)


def corp_mark(size, color_a=CRIMSON, color_b=(245, 245, 245), rot=0.0):
    """The company's mark: a closing iris of six blades inside a thin ring.

    An original device rather than anything borrowed from the games this story takes its
    names from. It suits the company better anyway -- an aperture narrowing on whatever is
    underneath it -- and it holds its silhouette from a 24px panel header up to the side of
    a tower, which the mark has to do here.
    """
    ss = 4                                     # supersample: the blades have long thin edges
    img = Image.new('RGBA', (size * ss, size * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = size * ss / 2
    r = size * ss * 0.46
    inner = r * 0.30
    twist = 32.0                               # rotation from outer edge to inner, in degrees
    span = 46.0                                # angular width of a blade; the rest is gap

    def polar(rad, deg):
        a = math.radians(deg - 90 + rot)
        return (c + rad * math.cos(a), c + rad * math.sin(a))

    d.ellipse([c - r, c - r, c + r, c + r], outline=color_b, width=max(1, int(size * ss * 0.035)))
    for k in range(6):
        a = k * 60.0
        d.polygon([polar(r * 0.88, a), polar(r * 0.88, a + span),
                   polar(inner, a + span + twist), polar(inner, a + twist)],
                  fill=color_a)
    d.ellipse([c - inner * 0.52, c - inner * 0.52, c + inner * 0.52, c + inner * 0.52], fill=color_b)
    return img.resize((size, size), Image.LANCZOS)


def _wrap(draw, text, fnt, width):
    words, lines, cur = text.split(), [], ''
    for w in words:
        test = (cur + ' ' + w).strip()
        if draw.textlength(test, font=fnt) <= width:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def system_panel(title, lines):
    """Return (panel RGBA, list of text-row boxes (x0, y0, x1, y1) for the type-on reveal, text RGBA layer)."""
    ft = font('mono_b', 40)
    fl = font('mono', 32)
    pad, width = 50, 1060
    probe = ImageDraw.Draw(Image.new('RGBA', (10, 10)))
    rows = []
    for ln in lines:
        if ':' in ln and not ln.endswith(':'):
            k, v = (x.strip() for x in ln.split(':', 1))
            # a "Label: value" line is a ruled row when label and value fit side by side with
            # room for the leader dots; judged in pixels, since Shop items run long
            if len(k) <= 44 and probe.textlength(k, font=fl) + probe.textlength(v, font=fl) + 60 <= width - 2 * pad:
                rows.append(('kv', k, v))
                continue
        for w in _wrap(probe, ln, fl, width - 2 * pad):
            rows.append(('tx', w, ''))
    title_lines = _wrap(probe, title.upper(), ft, width - 2 * pad - 40)
    row_h = 48
    h = pad + len(title_lines) * 52 + 26 + len(rows) * row_h + pad
    panel = Image.new('RGBA', (width, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(panel)
    d.rounded_rectangle([0, 0, width - 1, h - 1], radius=10, fill=GLASS + (205,), outline=CRIMSON + (235,), width=2)
    d.rounded_rectangle([6, 6, width - 7, h - 7], radius=7, outline=(255, 255, 255, 38), width=1)
    # corner ticks
    for (x, y, dx, dy) in [(0, 0, 1, 1), (width - 1, 0, -1, 1), (0, h - 1, 1, -1), (width - 1, h - 1, -1, -1)]:
        d.line([(x, y + dy * 28), (x, y), (x + dx * 28, y)], fill=CRIMSON + (255,), width=4)
    text = Image.new('RGBA', (width, h), (0, 0, 0, 0))
    td = ImageDraw.Draw(text)
    boxes = []
    y = pad - 4
    mark = corp_mark(36)
    panel.alpha_composite(mark, (pad, y + 7))
    for tl in title_lines:
        td.text((pad + 48, y), tl, font=ft, fill=WHITE + (255,))
        boxes.append((pad + 44, y, pad + 44 + int(td.textlength(tl, font=ft)) + 4, y + 44))
        y += 52
    d.line([(pad, y + 8), (width - pad, y + 8)], fill=CRIMSON + (180,), width=2)
    y += 26
    for kind, a, b in rows:
        if kind == 'kv':
            td.text((pad, y), a, font=fl, fill=(190, 196, 206, 255))
            vw = td.textlength(b, font=font('mono_b', 32))
            # dotted leader
            ax = pad + td.textlength(a, font=fl) + 12
            bx = width - pad - vw - 12
            xx = ax
            while xx < bx:
                td.text((xx, y), '.', font=fl, fill=(120, 126, 136, 200))
                xx += 16
            td.text((width - pad - vw, y), b, font=font('mono_b', 32), fill=WHITE + (255,))
            boxes.append((pad, y, width - pad, y + row_h))
        else:
            td.text((pad, y), a, font=fl, fill=(222, 226, 232, 255))
            boxes.append((pad, y, pad + int(td.textlength(a, font=fl)) + 4, y + row_h))
        y += row_h
    glow = panel.filter(ImageFilter.GaussianBlur(14))
    out = Image.new('RGBA', (width + 60, h + 60), (0, 0, 0, 0))
    ga = np.array(glow)
    ga[..., :3] = np.array(CRIMSON)
    ga[..., 3] = (ga[..., 3] * 0.35).astype(np.uint8)
    out.alpha_composite(Image.fromarray(ga), (30, 30))
    out.alpha_composite(panel, (30, 30))
    txt = Image.new('RGBA', out.size, (0, 0, 0, 0))
    txt.alpha_composite(text, (30, 30))
    boxes = [(x0 + 30, y0 + 30, x1 + 30, y1 + 30) for x0, y0, x1, y1 in boxes]
    return out, boxes, txt


def loc_caption(text):
    f = font('cond_b', 34)
    probe = ImageDraw.Draw(Image.new('RGBA', (10, 10)))
    w = int(probe.textlength(text.upper(), font=f)) + 60
    img = Image.new('RGBA', (w + 40, 90), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 18, 5, 72], fill=CRIMSON + (255,))
    # soft shadow
    sh = Image.new('RGBA', img.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).text((24, 26), text.upper(), font=f, fill=(0, 0, 0, 200))
    img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(4)))
    d.text((22, 24), text.upper(), font=f, fill=WHITE + (255,))
    return img, (22, 24, w, 70)


def _center(d, y, text, fnt, fill, width=W, spacing=0):
    if spacing:
        total = sum(d.textlength(ch, font=fnt) + spacing for ch in text) - spacing
        x = (width - total) / 2
        for ch in text:
            d.text((x, y), ch, font=fnt, fill=fill)
            x += d.textlength(ch, font=fnt) + spacing
        return
    tw = d.textlength(text, font=fnt)
    d.text(((width - tw) / 2, y), text, font=fnt, fill=fill)


def _dark_bg(seed=0, tint=(40, 6, 10)):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2)
    base = np.clip(1 - r, 0, 1) ** 1.6
    img = np.zeros((H, W, 3), np.float32)
    for k in range(3):
        img[..., k] = 6 + tint[k] * base
    img += rng.normal(0, 2.2, img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).convert('RGBA')


def movement_card(num, title):
    img = _dark_bg(1)
    d = ImageDraw.Draw(img)
    mark = corp_mark(180)
    img.alpha_composite(mark, ((W - 180) // 2, 250))
    words = {'I': 'ONE', 'II': 'TWO', 'III': 'THREE', 'IV': 'FOUR', 'V': 'FIVE', 'VI': 'SIX'}[num]
    _center(d, 500, f'PART {words}', font('cond_b', 40), CRIMSON + (255,), spacing=14)
    _center(d, 570, title.upper(), font('serif_b', 92), WHITE + (255,), spacing=4)
    d.line([(W / 2 - 220, 700), (W / 2 + 220, 700)], fill=(200, 200, 200, 120), width=2)
    return img.convert('RGB')


def chapter_card(num, title, movement=None):
    """The chapter's title card. A chapter that opens a Part also names the Part, but only as
    a small label above the chapter number, so the chapter title stays the one line set in
    title type and the card reads as one title, not two."""
    img = _dark_bg(num + 10, tint=(28, 8, 10))
    d = ImageDraw.Draw(img)
    top = 400
    if movement:
        pnum, ptitle = movement
        words = {'I': 'ONE', 'II': 'TWO', 'III': 'THREE', 'IV': 'FOUR', 'V': 'FIVE', 'VI': 'SIX'}[pnum]
        mark = corp_mark(96)
        img.alpha_composite(mark, ((W - 96) // 2, 196))
        _center(d, 322, f'PART {words}  ·  {ptitle.upper()}', font('cond_b', 30), (196, 196, 202, 255), spacing=10)
        d.line([(W / 2 - 60, 382), (W / 2 + 60, 382)], fill=(200, 200, 200, 90), width=1)
        top = 420
    _center(d, top, f'CHAPTER {num}', font('cond_b', 38), CRIMSON + (255,), spacing=12)
    f = font('serif', 84)
    lines = _wrap(d, title, f, 1500)
    y = top + 70
    for ln in lines:
        _center(d, y, ln, f, WHITE + (255,))
        y += 100
    d.line([(W / 2 - 160, y + 30), (W / 2 + 160, y + 30)], fill=CRIMSON + (200,), width=3)
    return img.convert('RGB')


def main_title():
    img = _dark_bg(99, tint=(55, 5, 10))
    d = ImageDraw.Draw(img)
    mark = corp_mark(260)
    img.alpha_composite(mark, ((W - 260) // 2, 170))
    _center(d, 470, 'UMBRELLA', font('serif_b', 150), WHITE + (255,), spacing=22)
    _center(d, 670, 'WHAT IF HITLER WAS REBORN AS ALBERT WESKER', font('cond_b', 40), (210, 210, 215, 255), spacing=6)
    _center(d, 730, 'AND GIVEN A SYSTEM TO BUILD AN EMPIRE?', font('cond', 34), CRIMSON + (255,), spacing=6)
    return img.convert('RGB')


def placeholder(prompt, seed=0):
    """Test image when no generated image exists yet: moody gradient plus the prompt."""
    rng = np.random.default_rng(seed)
    tint = tuple(int(x) for x in rng.integers(20, 90, 3))
    img = _dark_bg(seed, tint=tint)
    d = ImageDraw.Draw(img)
    f = font('serif', 44)
    lines = _wrap(d, prompt, f, 1500)[:6]
    y = H / 2 - len(lines) * 30
    for ln in lines:
        _center(d, y, ln, f, (220, 220, 220, 255))
        y += 60
    return img.convert('RGB')


if __name__ == '__main__':
    os.makedirs('/tmp/claude-0/gfx', exist_ok=True)
    p, boxes, txt = system_panel('Status', ['Level: 1', 'Strength: 34', 'Speed: 38', 'Intelligence: 22', 'Charisma: 24'])
    p.alpha_composite(txt)
    p.save('/tmp/claude-0/gfx/panel.png')
    movement_card('I', 'The Garage Empire').save('/tmp/claude-0/gfx/movement.png')
    chapter_card(1, 'Two Deaths and a Chair').save('/tmp/claude-0/gfx/chapter.png')
    main_title().save('/tmp/claude-0/gfx/title.png')
    loc_caption('Wilmington, Delaware — 14 September 2026, 3:12 a.m.')[0].save('/tmp/claude-0/gfx/loc.png')
    print('ok')
