"""Sound-effect library: recorded sources where good ones exist, synthesis for the rest.

Run `python3 pipeline/sfx_lib.py` to (re)build every missing file in assets/sfx.
One-shots are peak-normalized to -1 dBFS; ambiences are loop-ready and RMS-normalized.
"""
import os
import glob
import sys
import numpy as np
from dsp import *  # noqa: F401,F403
from pedalboard import PitchShift, Distortion, Chorus, Delay

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'assets', 'sfx')
MG = '/usr/share/games/megaglest/tilesets'
OA = '/opt/sfxsrc/oa/sound'
WZ = '/opt/sfxsrc/wz/audio'
REG = {}


def sfx(fn):
    REG[fn.__name__] = fn
    return fn


def rng(seed):
    return np.random.default_rng(seed)


def amb_norm(x, db=-20.0):
    cur = rms_db(x)
    return (x * 10 ** ((db - cur) / 20)).astype(np.float32)


def src(path):
    return load(path) if os.path.exists(path) else None


# ---------------------------------------------------------------- building blocks

def thump(freq=55, sec=0.4, tau=0.12, sweep_to=None):
    f1 = sweep_to or freq * 0.6
    return (sine_sweep(sec, freq, f1) * expdecay(sec, tau)).astype(np.float32)


def noise_burst(sec, tau, lo=None, hi=None, seed=0):
    x = white(sec, rng(seed)) * expdecay(sec, tau)
    if lo and hi:
        x = bp(x, lo, hi)
    elif lo:
        x = hp(x, lo)
    elif hi:
        x = lp(x, hi)
    return x.astype(np.float32)


def gunshot(seed=0, muffled=False, distance=0.0):
    r = rng(seed)
    crack = noise_burst(0.012, 0.003, lo=1500, seed=seed) * 1.2
    body = noise_burst(0.25, 0.045, hi=5000, seed=seed + 1)
    low = thump(90 + r.uniform(-10, 10), 0.35, 0.08, 38)
    x = np.zeros(int(0.4 * SR), np.float32)
    x[:len(crack)] += crack
    x[:len(body)] += body * 0.9
    x[:len(low)] += low * 1.1
    x = np.tanh(x * 2.5) * 0.8
    if muffled:
        x = lp(x, 900)
        y = reverb(x, room=0.55, wet=0.45, dry=0.7, damp=0.7, tail=1.5)
    else:
        y = reverb(x, room=0.75 + distance * 0.2, wet=0.25 + distance * 0.4, dry=1.0 - distance * 0.6, damp=0.4, tail=2.0)
        if distance:
            y = lp(y, 5000 - 3500 * distance)
    return y


def boom(sec=4.0, seed=0, fc=900, sub=45):
    r = rng(seed)
    n = brown(sec, r) * expdecay(sec, sec * 0.25)
    n = lp(n, fc)
    s = thump(sub, sec, sec * 0.18, sub * 0.5) * 1.3
    crack = noise_burst(0.08, 0.02, hi=6000, seed=seed + 7) * 0.8
    x = n * 1.2 + s
    x[:len(crack)] += crack
    # debris crackle
    deb = np.zeros_like(x)
    for _ in range(80):
        i = int(r.uniform(0.1, sec * 0.7) * SR)
        g = noise_burst(0.02, 0.004, lo=1500, hi=9000, seed=int(r.integers(1e6)))
        deb[i:i + len(g)] += g[:len(deb) - i] * r.uniform(0.05, 0.25) * np.exp(-i / SR / (sec * 0.3))
    x = np.tanh((x + deb) * 1.8)
    return reverb(x, room=0.9, wet=0.35, dry=0.9, damp=0.5, tail=2.0)


def clap(seed):
    x = noise_burst(0.03, 0.006, lo=800, hi=3500, seed=seed)
    return x


def footstep_samples():
    files = sorted(glob.glob(f'{OA}/player/footsteps/boot*.wav'))
    return [load(f) for f in files]


# ---------------------------------------------------------------- ambiences

@sfx
def amb_rain_heavy():
    r = rng(11)
    sec = 60
    bed = lp(pink(sec, r), 7000) * 0.5 + hp(white(sec, r), 3000) * 0.08
    buf = np.stack([bed, np.roll(bed, 9000)])
    # droplets
    for _ in range(9000):
        i = int(r.uniform(0, sec - 0.05) * SR)
        d = noise_burst(0.01, 0.0015, lo=2000, hi=11000, seed=int(r.integers(1e6)))
        ch = r.integers(2)
        buf[ch, i:i + len(d)] += d * r.uniform(0.05, 0.25)
    for path in [f'{MG}/evergreen/sounds/rain.ogg', f'{MG}/autumn/sounds/rain.ogg', f'{MG}/desert2/sounds/rain.wav']:
        s = src(path)
        if s is None:
            continue
        s = s / (np.abs(s).max() + 1e-9)
        pos = 0.0
        while pos < sec:
            place(buf, fade(s, 1.0, 1.0), pos, 0.9)
            pos += s.shape[1] / SR - 1.5
    buf = lp(buf, 9000)
    return amb_norm(make_loop(buf, 4.0), -20)


@sfx
def amb_rain_window():
    x = amb_rain_heavy()
    x = lp(x, 2500)
    r = rng(12)
    n = x.shape[1]
    # occasional heavier drops tapping glass
    for _ in range(600):
        i = int(r.uniform(0, n / SR - 0.1) * SR)
        d = fm_bell(r.uniform(1800, 3200), 0.05, index=1, tau=0.01) * 0.2
        x[:, i:i + len(d)] += d
    return amb_norm(x, -21)


@sfx
def amb_office_hum():
    sec = 40
    tt = t(sec)
    hum = sum(a * np.sin(2 * np.pi * f * tt) for f, a in [(60, 0.5), (120, 0.35), (180, 0.15), (240, 0.08)])
    buzz = bp(white(sec, rng(21)), 7000, 9000) * (0.5 + 0.5 * np.sin(2 * np.pi * 120 * tt)) * 0.15
    air = lp(brown(sec, rng(22)), 350) * 0.6
    x = hum * 0.25 + buzz + air
    return amb_norm(make_loop(stereo(x.astype(np.float32), 0.3, 12), 3), -26)


@sfx
def amb_bunker():
    sec = 60
    r = rng(31)
    rumble = lp(brown(sec, r), 110) * (0.7 + 0.3 * np.sin(2 * np.pi * 0.07 * t(sec)))
    buf = stereo(rumble.astype(np.float32), 0.5, 20)
    for k in range(9):
        at = r.uniform(1, sec - 6)
        b = boom(4.0, seed=100 + k, fc=180, sub=38)
        place(buf, lp(b, 220), at, r.uniform(0.25, 0.6))
    for _ in range(400):
        i = int(r.uniform(0, sec - 0.02) * SR)
        g = noise_burst(0.01, 0.002, lo=3000, hi=9000, seed=int(r.integers(1e6)))
        buf[r.integers(2), i:i + len(g)] += g * 0.03
    return amb_norm(make_loop(buf, 4), -22)


@sfx
def amb_fire_volcano():
    sec = 50
    r = rng(41)
    roar = lp(brown(sec, r), 500) * (0.6 + 0.4 * lp(np.abs(white(sec, r)), 2))
    sub = np.sin(2 * np.pi * 32 * t(sec)) * 0.3
    buf = stereo((roar + sub).astype(np.float32), 0.6, 18)
    for _ in range(3000):
        i = int(r.uniform(0, sec - 0.03) * SR)
        g = noise_burst(0.02, 0.003, lo=1200, hi=7000, seed=int(r.integers(1e6)))
        buf[r.integers(2), i:i + len(g)] += g * r.uniform(0.03, 0.15)
    return amb_norm(make_loop(buf, 4), -19)


@sfx
def amb_lab():
    sec = 45
    tt = t(sec)
    comp = (np.sin(2 * np.pi * 98 * tt) * 0.3 + np.sin(2 * np.pi * 100.7 * tt) * 0.3 + np.sin(2 * np.pi * 196 * tt) * 0.1)
    vent = lp(pink(sec, rng(51)), 1800) * 0.35
    x = stereo((comp * 0.3 + vent).astype(np.float32), 0.5, 15)
    r = rng(52)
    for k in range(5):  # distant equipment beeps
        at = r.uniform(2, sec - 2)
        b = fm_bell(2000, 0.12, index=0.2, tau=0.05) * 0.05
        place(x, reverb(b, 0.5, 0.4, 0.6), at)
    return amb_norm(make_loop(x, 3), -25)


@sfx
def amb_basement():
    sec = 48
    tt = t(sec)
    cyc = 0.5 + 0.5 * np.clip(np.sin(2 * np.pi * tt / 16), -0.2, 1)
    comp = (np.sin(2 * np.pi * 50 * tt) * 0.4 + np.sin(2 * np.pi * 100 * tt) * 0.2) * cyc
    air = lp(brown(sec, rng(61)), 250)
    x = stereo((comp * 0.4 + air * 0.6).astype(np.float32), 0.5, 25)
    r = rng(62)
    for _ in range(14):  # drips
        at = r.uniform(0, sec - 1)
        d = fm_bell(r.uniform(900, 1500), 0.2, index=0.5, tau=0.04) * 0.08
        place(x, reverb(pan(d, r.uniform(-0.7, 0.7))[0], 0.7, 0.5, 0.5), at)
    return amb_norm(make_loop(x, 3), -24)


@sfx
def amb_cryo():
    """The cryo-cradle: a compressor that sounds like breathing."""
    sec = 48
    tt = t(sec)
    breath = (0.5 - 0.5 * np.cos(2 * np.pi * tt / 6.0)) ** 2
    inhale = bp(pink(sec, rng(71)), 300, 1800) * breath
    hum = (np.sin(2 * np.pi * 41 * tt) + 0.5 * np.sin(2 * np.pi * 82.3 * tt)) * 0.25
    hiss = hp(white(sec, rng(72)), 6000) * 0.02
    x = stereo((inhale * 0.8 + hum + hiss).astype(np.float32), 0.6, 20)
    x = reverb(x, 0.7, 0.3, 0.8, damp=0.6)[:, :int(sec * SR)]
    return amb_norm(make_loop(x, 3), -23)


@sfx
def amb_wind():
    sec = 60
    r = rng(81)
    base = pink(sec, r)
    cf = 400 + 300 * lp(np.abs(white(sec, r)), 0.3) / 0.05
    x = bp(base, 150, 1200) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.05 * t(sec)))
    buf = stereo(x.astype(np.float32), 0.6, 30)
    s = src(f'{MG}/winter/sounds/wind.ogg')
    if s is not None:
        s = s / (np.abs(s).max() + 1e-9)
        pos = 0.0
        while pos < sec:
            place(buf, fade(s, 1.5, 1.5), pos, 0.8)
            pos += s.shape[1] / SR - 2
    return amb_norm(make_loop(buf, 4), -21)


@sfx
def amb_night_city():
    sec = 60
    r = rng(91)
    traffic = lp(pink(sec, r), 900) * (0.7 + 0.3 * np.sin(2 * np.pi * 0.03 * t(sec)))
    buf = stereo(traffic.astype(np.float32), 0.7, 25)
    for k in range(6):  # far cars
        at = r.uniform(0, sec - 8)
        c = car_pass(seed=500 + k, dist=0.9)
        place(buf, c, at, 0.35)
    return amb_norm(make_loop(buf, 4), -24)


@sfx
def amb_ship_hum():
    sec = 50
    tt = t(sec)
    drone = sum(np.sin(2 * np.pi * f * tt + p) * a for f, p, a in [(36, 0, 0.5), (36.4, 1, 0.5), (72.1, 2, 0.2), (108, 0.3, 0.1)])
    puls = 0.8 + 0.2 * np.sin(2 * np.pi * 0.25 * tt)
    air = lp(pink(sec, rng(101)), 600) * 0.4
    x = stereo((drone * puls * 0.4 + air).astype(np.float32), 0.7, 20)
    return amb_norm(make_loop(x, 3), -22)


@sfx
def amb_space():
    sec = 60
    tt = t(sec)
    sub = np.sin(2 * np.pi * 28 * tt) * 0.4
    shimmer = bp(white(sec, rng(111)), 5000, 9000) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * tt)) * 0.05
    x = stereo((sub + shimmer).astype(np.float32), 0.8, 30)
    x = reverb(x, 0.95, 0.5, 0.6)[:, :int(sec * SR)]
    return amb_norm(make_loop(x, 4), -24)


# ---------------------------------------------------------------- one-shots

@sfx
def sfx_gunshot_single():
    return normalize(gunshot(1))


@sfx
def sfx_gunshot_muffled():
    return normalize(gunshot(2, muffled=True))


@sfx
def sfx_gunshot_distant():
    return normalize(gunshot(3, distance=0.8))


@sfx
def sfx_gunfire_burst():
    buf = np.zeros((2, int(3.5 * SR)), np.float32)
    r = rng(5)
    at = 0.0
    for k in range(14):
        place(buf, gunshot(10 + k), at, r.uniform(0.7, 1.0))
        at += r.uniform(0.075, 0.095)
    return normalize(buf)


@sfx
def sfx_gunfire_exchange():
    buf = np.zeros((2, int(7 * SR)), np.float32)
    r = rng(6)
    for k in range(30):
        g = gunshot(40 + k, distance=r.uniform(0, 0.7))
        place(buf, g * np.array([[r.uniform(0.5, 1)], [r.uniform(0.5, 1)]]), r.uniform(0, 5), r.uniform(0.4, 1))
    return normalize(buf)


@sfx
def sfx_explosion():
    x = boom(5.0, seed=7)
    s = src(f'{WZ}/sfx/explons/lrgexpl.ogg')
    if s is not None:
        place(x, s / (np.abs(s).max() + 1e-9), 0.0, 0.6)
    return normalize(x)


@sfx
def sfx_explosion_distant():
    x = boom(6.0, seed=8, fc=400, sub=35)
    x = lp(x, 700)
    return normalize(reverb(x, 0.95, 0.5, 0.6, tail=2))


@sfx
def sfx_artillery_distant():
    x = boom(4.0, seed=9, fc=250, sub=40)
    return normalize(reverb(lp(x, 300), 0.95, 0.5, 0.7, tail=2), -3)


@sfx
def sfx_rocket():
    sec = 1.6
    w = bp(white(sec, rng(13)), 600, 6000) * adsr(int(sec * SR), 0.05, 0.3, 0.7, 0.9)
    tone = sine_sweep(sec, 300, 1400) * 0.1 * adsr(int(sec * SR), 0.1, 0.2, 0.6, 0.8)
    x = pan((w + tone).astype(np.float32), 0)
    # sweep across stereo field
    p = np.linspace(-0.8, 0.8, x.shape[1])
    x = np.stack([x[0] * np.cos((p + 1) * np.pi / 4), x[1] * np.sin((p + 1) * np.pi / 4)])
    return normalize(reverb(x, 0.6, 0.25, 0.9))


@sfx
def sfx_thunder():
    sec = 8
    r = rng(14)
    crack = noise_burst(0.3, 0.05, lo=200, hi=8000, seed=14)
    rum = lp(brown(sec, r), 250) * expdecay(sec, 2.5) * (0.5 + 0.5 * lp(np.abs(white(sec, r)), 3) / 0.03)
    x = rum.astype(np.float32)
    x[:len(crack)] += crack * 0.5
    return normalize(reverb(np.tanh(x * 1.5), 0.95, 0.4, 0.8))


@sfx
def sfx_heartbeat():
    buf = np.zeros(int(1.6 * SR), np.float32)
    for at, g in [(0.0, 1.0), (0.28, 0.7)]:
        b = lp(mixp(thump(62, 0.25, 0.05, 40), thump(95, 0.1, 0.02, 60) * 0.3)[0], 180)
        i = int(at * SR)
        buf[i:i + len(b)] += b * g
    return normalize(reverb(buf, 0.3, 0.15, 0.9))


@sfx
def sfx_heartbeat_slow():
    buf = np.zeros((2, int(5 * SR)), np.float32)
    hb = sfx_heartbeat()
    for k in range(4):
        place(buf, hb, k * 1.15, 1.0 - k * 0.15)
    return normalize(buf)


@sfx
def sfx_breath_gasp():
    sec = 0.7
    n = int(sec * SR)
    env = adsr(n, 0.12, 0.2, 0.4, 0.3)
    x = bp(pink(sec, rng(15)), 700, 3500) * env
    x += bp(white(sec, rng(16)), 2500, 6000) * env * 0.2
    return normalize(reverb(x, 0.2, 0.1, 0.95), -6)


@sfx
def sfx_mug_catch():
    x = sum(fm_bell(f, 0.5, index=0.3, tau=tt) * a for f, tt, a in [(2150, 0.12, 0.5), (3480, 0.08, 0.3), (5230, 0.05, 0.2)])
    slosh = bp(pink(0.3, rng(17)), 300, 1500) * adsr(int(0.3 * SR), 0.03, 0.1, 0.3, 0.15) * 0.4
    y = np.zeros(int(0.6 * SR), np.float32)
    y[:len(slosh)] += slosh
    y[int(0.02 * SR):int(0.02 * SR) + len(x)] += x[:len(y) - int(0.02 * SR)] * 0.5
    return normalize(reverb(y, 0.3, 0.2, 0.9), -4)


@sfx
def sfx_door_knock():
    buf = np.zeros(int(1.2 * SR), np.float32)
    for k, at in enumerate([0.0, 0.2, 0.4]):
        body = bp(noise_burst(0.08, 0.012, seed=18 + k), 150, 1200)
        res = np.sin(2 * np.pi * 170 * t(0.08)) * expdecay(0.08, 0.02) * 0.6
        i = int(at * SR)
        buf[i:i + len(body)] += body + res
    return normalize(reverb(buf, 0.35, 0.2, 0.9))


@sfx
def sfx_door_open():
    buf = np.zeros(int(1.5 * SR), np.float32)
    click = noise_burst(0.02, 0.004, lo=1500, hi=6000, seed=19)
    buf[:len(click)] += click
    sw = bp(pink(1.0, rng(20)), 200, 900) * adsr(int(SR), 0.3, 0.3, 0.3, 0.4) * 0.3
    buf[int(0.15 * SR):int(0.15 * SR) + len(sw)] += sw
    thud = thump(80, 0.2, 0.04, 50) * 0.3
    buf[int(1.1 * SR):int(1.1 * SR) + len(thud)] += thud
    return normalize(reverb(buf, 0.4, 0.25, 0.9))


@sfx
def sfx_door_slam():
    x = lp(mixp(thump(70, 0.5, 0.08, 45), noise_burst(0.15, 0.02, hi=2500, seed=21) * 0.8), 3000)
    return normalize(reverb(x, 0.7, 0.35, 0.9, tail=1))


@sfx
def sfx_footsteps():
    steps = footstep_samples()
    buf = np.zeros((2, int(4.5 * SR)), np.float32)
    r = rng(22)
    for k in range(7):
        s = steps[k % len(steps)] if steps else stereo(lp(noise_burst(0.1, 0.02, seed=k), 2000))
        place(buf, s, 0.1 + k * 0.55 + r.uniform(-0.02, 0.02), 1.0 - k * 0.11)
    return normalize(reverb(lp(buf, 6000), 0.55, 0.3, 0.9))


@sfx
def sfx_footsteps_approach():
    steps = footstep_samples()
    buf = np.zeros((2, int(4.5 * SR)), np.float32)
    for k in range(7):
        s = steps[k % len(steps)] if steps else stereo(lp(noise_burst(0.1, 0.02, seed=k), 2000))
        place(buf, s, 0.1 + k * 0.55, 0.35 + k * 0.11)
    return normalize(reverb(lp(buf, 6000), 0.55, 0.3, 0.9))


@sfx
def sfx_keyboard():
    sec = 4.0
    buf = np.zeros(int(sec * SR), np.float32)
    r = rng(23)
    at = 0.05
    while at < sec - 0.1:
        k = noise_burst(0.015, 0.003, lo=1500, hi=7000, seed=int(r.integers(1e6)))
        k += np.sin(2 * np.pi * r.uniform(1800, 3000) * t(0.015)) * expdecay(0.015, 0.002) * 0.3
        i = int(at * SR)
        buf[i:i + len(k)] += k * r.uniform(0.5, 1.0)
        at += r.uniform(0.07, 0.16) if r.random() > 0.08 else r.uniform(0.3, 0.6)
    return normalize(reverb(buf, 0.25, 0.15, 0.9), -3)


@sfx
def sfx_mouse_click():
    buf = np.zeros(int(0.4 * SR), np.float32)
    for at in (0.0, 0.12):
        c = noise_burst(0.01, 0.0015, lo=2000, hi=8000, seed=int(at * 100))
        i = int(at * SR)
        buf[i:i + len(c)] += c
    return normalize(buf, -4)


@sfx
def sfx_phone_buzz():
    sec = 1.4
    tt = t(sec)
    gate = ((tt % 0.7) < 0.42).astype(np.float32)
    x = np.sign(np.sin(2 * np.pi * 160 * tt)) * 0.3 + np.sin(2 * np.pi * 160 * tt) * 0.5
    x = lp(x * gate, 900)
    return normalize(reverb(x, 0.3, 0.2, 0.9), -3)


@sfx
def sfx_phone_ring():
    sec = 3.0
    tt = t(sec)
    gate = ((tt % 1.5) < 0.9).astype(np.float32)
    trill = (np.sin(2 * np.pi * 20 * tt) > 0).astype(np.float32)
    x = (np.sin(2 * np.pi * 1300 * tt) + np.sin(2 * np.pi * 1600 * tt)) * gate * (0.5 + 0.5 * trill)
    return normalize(reverb(lp(x, 4000), 0.4, 0.25, 0.8), -4)


@sfx
def sfx_system_open():
    sec = 2.2
    buf = np.zeros(int(sec * SR), np.float32)
    for k, f in enumerate([880, 1318.5, 1760, 2637]):
        b = fm_bell(f, 1.6, index=1.2, ratio=2.0, tau=0.5) * (0.5 - k * 0.07)
        i = int(k * 0.06 * SR)
        buf[i:i + len(b)] += b[:len(buf) - i]
    air = bp(white(sec, rng(24)), 3000, 10000) * adsr(int(sec * SR), 0.25, 0.3, 0.2, 1.2) * 0.12
    x = buf + air
    return normalize(reverb(x, 0.85, 0.4, 0.7, width=1.0), -2)


@sfx
def sfx_system_chime():
    sec = 2.4
    buf = np.zeros(int(sec * SR), np.float32)
    for k, f in enumerate([1318.5, 987.8, 1661.2]):
        b = fm_bell(f, 1.8, index=0.9, ratio=3.5, tau=0.6) * 0.5
        i = int(k * 0.14 * SR)
        buf[i:i + len(b)] += b[:len(buf) - i]
    return normalize(reverb(buf, 0.8, 0.35, 0.75), -2)


@sfx
def sfx_level_up():
    sec = 3.2
    buf = np.zeros(int(sec * SR), np.float32)
    notes = [523.25, 659.25, 783.99, 1046.5, 1318.5, 1568.0]
    for k, f in enumerate(notes):
        b = fm_bell(f, 2.2, index=1.0, ratio=2.0, tau=0.7) * 0.4
        i = int(k * 0.07 * SR)
        buf[i:i + len(b)] += b[:len(buf) - i]
    swell = sine_sweep(sec, 65, 131) * adsr(int(sec * SR), 0.3, 0.5, 0.4, 2.0) * 0.4
    x = buf + swell
    return normalize(reverb(x, 0.9, 0.45, 0.7), -1.5)


@sfx
def sfx_points():
    sec = 1.3
    buf = np.zeros(int(sec * SR), np.float32)
    for k, f in enumerate([1760, 2217.5, 2637]):
        b = fm_bell(f, 0.9, index=0.6, ratio=2.0, tau=0.25) * 0.4
        i = int(k * 0.045 * SR)
        buf[i:i + len(b)] += b[:len(buf) - i]
    return normalize(reverb(buf, 0.6, 0.3, 0.8), -4)


@sfx
def sfx_gacha_spin():
    sec = 3.5
    buf = np.zeros(int(sec * SR), np.float32)
    at, gap = 0.0, 0.18
    k = 0
    while at < sec - 0.1:
        tick = fm_bell(2400 + 40 * k, 0.05, index=0.3, tau=0.01)
        i = int(at * SR)
        buf[i:i + len(tick)] += tick * 0.5
        at += gap
        gap = max(0.035, gap * 0.9)
        k += 1
    riser = bp(white(sec, rng(25)), 1500, 8000) * np.linspace(0, 1, int(sec * SR)) ** 2 * 0.25
    tone = sine_sweep(sec, 110, 440) * np.linspace(0, 1, int(sec * SR)) ** 2 * 0.2
    return normalize(reverb(buf + riser + tone, 0.6, 0.3, 0.8), -2)


@sfx
def sfx_gacha_gold():
    sec = 5.0
    buf = np.zeros(int(sec * SR), np.float32)
    hit = boom(3.0, seed=26, fc=300, sub=48)
    for f in [523.25, 659.25, 783.99, 987.77, 1318.5, 2093]:
        b = fm_bell(f, 4.0, index=1.4, ratio=2.01, tau=1.4) * 0.22
        buf[:len(b)] += b
    x = stereo(buf, 0.5, 15)
    place(x, hit, 0.0, 0.8)
    return normalize(reverb(x, 0.95, 0.5, 0.7), -1)


@sfx
def sfx_gacha_common():
    b = fm_bell(660, 1.0, index=0.4, tau=0.3) * 0.5 + fm_bell(990, 1.0, index=0.3, tau=0.25) * 0.3
    return normalize(reverb(b, 0.5, 0.25, 0.8), -5)


@sfx
def sfx_gacha_rare():
    buf = np.zeros(int(2.2 * SR), np.float32)
    for k, f in enumerate([783.99, 987.77, 1174.7]):
        b = fm_bell(f, 1.8, index=0.9, tau=0.6) * 0.4
        i = int(k * 0.08 * SR)
        buf[i:i + len(b)] += b[:len(buf) - i]
    return normalize(reverb(buf, 0.75, 0.35, 0.8), -3)


@sfx
def sfx_whoosh():
    sec = 0.9
    n = int(sec * SR)
    env = np.sin(np.linspace(0, np.pi, n)) ** 2
    x = bp(pink(sec, rng(27)), 300, 5000) * env
    p = np.linspace(-0.7, 0.7, n)
    y = np.stack([x * np.cos((p + 1) * np.pi / 4), x * np.sin((p + 1) * np.pi / 4)])
    return normalize(reverb(y, 0.5, 0.2, 0.9), -3)


@sfx
def sfx_impact():
    return normalize(boom(4.0, seed=28, fc=1500, sub=50))


@sfx
def sfx_riser():
    sec = 4.0
    n = int(sec * SR)
    ramp = np.linspace(0, 1, n) ** 2.5
    x = bp(white(sec, rng(29)), 800, 9000) * ramp * 0.5 + sine_sweep(sec, 80, 640) * ramp * 0.3
    return normalize(reverb(x, 0.8, 0.35, 0.8), -2)


@sfx
def sfx_glass_break():
    sec = 1.5
    buf = np.zeros(int(sec * SR), np.float32)
    r = rng(30)
    for _ in range(120):
        at = r.uniform(0, 0.6) ** 2
        f = r.uniform(2500, 9000)
        b = fm_bell(f, 0.3, index=2, ratio=r.uniform(1.3, 2.7), tau=r.uniform(0.02, 0.1)) * r.uniform(0.1, 0.4)
        i = int(at * SR)
        buf[i:i + len(b)] += b[:len(buf) - i]
    buf[:int(0.05 * SR)] += noise_burst(0.05, 0.01, lo=1500, seed=30)
    return normalize(reverb(buf, 0.5, 0.3, 0.9))


@sfx
def sfx_camera_shutter():
    buf = np.zeros(int(0.35 * SR), np.float32)
    for at in (0.0, 0.09):
        c = noise_burst(0.02, 0.004, lo=1500, hi=9000, seed=int(at * 1000) + 3)
        i = int(at * SR)
        buf[i:i + len(c)] += c
    return normalize(buf, -3)


@sfx
def sfx_camera_flashes():
    buf = np.zeros((2, int(4 * SR)), np.float32)
    r = rng(31)
    shot = sfx_camera_shutter()
    for _ in range(18):
        place(buf, pan(shot[0] if shot.ndim == 2 else shot, r.uniform(-0.9, 0.9)), r.uniform(0, 3.6), r.uniform(0.3, 1))
    return normalize(reverb(buf, 0.5, 0.25, 0.9))


@sfx
def sfx_applause():
    sec = 7.0
    buf = np.zeros((2, int(sec * SR)), np.float32)
    r = rng(32)
    dens = adsr(int(sec * SR), 0.8, 1.0, 0.8, 2.5)
    for _ in range(2600):
        at = r.uniform(0, sec - 0.05)
        if r.random() > dens[int(at * SR)]:
            continue
        c = clap(int(r.integers(1e6))) * r.uniform(0.2, 1)
        place(buf, pan(c, r.uniform(-1, 1)), at)
    return normalize(reverb(buf, 0.8, 0.35, 0.85))


@sfx
def sfx_siren():
    sec = 8.0
    tt = t(sec)
    f = 950 + 350 * np.sin(2 * np.pi * tt / 3.5)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) + 0.3 * np.sin(2 * ph) + 0.15 * np.sin(3 * ph)
    env = adsr(len(x), 1.5, 0.5, 0.9, 3.0)
    x = lp(x * env, 3000) * 0.5
    return normalize(reverb(x, 0.85, 0.45, 0.6), -4)


def car_pass(seed=0, dist=0.3):
    r = rng(seed)
    sec = 6.0
    tt = t(sec)
    center = sec / 2
    rel = (tt - center) * 20
    dop = 1 + 0.08 * (-rel / np.sqrt(rel ** 2 + 30 ** 2))
    f0 = r.uniform(35, 55) * dop
    ph = 2 * np.pi * np.cumsum(f0) / SR
    eng = sum(np.sin(k * ph) / k for k in range(1, 8))
    tyre = lp(pink(sec, r), 1500) * 0.8
    amp = 1 / (1 + (rel / 12) ** 2)
    x = (eng * 0.3 + tyre) * amp
    x = lp(x, 4000 - 3000 * dist)
    p = np.clip(rel / 40, -1, 1)
    return np.stack([x * np.cos((p + 1) * np.pi / 4), x * np.sin((p + 1) * np.pi / 4)]).astype(np.float32)


@sfx
def sfx_car_pass():
    return normalize(reverb(car_pass(33), 0.4, 0.15, 0.9), -3)


@sfx
def sfx_car_door():
    x = lp(mixp(thump(75, 0.4, 0.06, 45) * 1.2, noise_burst(0.12, 0.02, hi=3000, seed=34) * 0.6), 3500)
    return normalize(reverb(x, 0.3, 0.15, 0.9), -3)


@sfx
def sfx_engine_start():
    sec = 4.0
    tt = t(sec)
    f = 30 + 25 * np.clip(tt / 0.8, 0, 1) - 8 * np.clip((tt - 1) / 2, 0, 1)
    ph = 2 * np.pi * np.cumsum(f) / SR
    eng = sum(np.sin(k * ph) / k for k in range(1, 9)) * adsr(len(tt), 0.3, 0.5, 0.7, 1.0)
    return normalize(reverb(lp(eng, 2500), 0.3, 0.15, 0.9), -4)


@sfx
def sfx_elevator_ding():
    b = fm_bell(1174.7, 2.0, index=0.3, ratio=2.0, tau=0.8) * 0.6
    return normalize(reverb(b, 0.5, 0.25, 0.85), -4)


@sfx
def sfx_elevator_descend():
    sec = 6.0
    tt = t(sec)
    hum = (np.sin(2 * np.pi * 55 * tt) * 0.4 + lp(pink(sec, rng(35)), 500)) * adsr(len(tt), 0.8, 0.5, 0.8, 1.5)
    clunk = thump(90, 0.3, 0.05, 50)
    x = hum.astype(np.float32)
    x[:len(clunk)] += clunk
    x[-len(clunk) - int(0.5 * SR):-int(0.5 * SR)] += clunk
    return normalize(reverb(x, 0.5, 0.25, 0.9), -3)


@sfx
def sfx_cryo_hiss():
    sec = 3.5
    n = int(sec * SR)
    hiss = hp(white(sec, rng(36)), 2500) * adsr(n, 0.02, 0.4, 0.6, 2.2) * 0.6
    clunk = thump(70, 0.4, 0.06, 40)
    x = hiss.astype(np.float32)
    x[:len(clunk)] += clunk
    return normalize(reverb(x, 0.7, 0.35, 0.8))


@sfx
def sfx_tyrant_growl():
    sec = 3.5
    tt = t(sec)
    r = rng(37)
    jitter = lp(r.standard_normal(len(tt)), 8) * 30
    f = 48 + jitter
    ph = 2 * np.pi * np.cumsum(f) / SR
    saw = sum(np.sin(k * ph) / k for k in range(1, 30))
    am = 0.6 + 0.4 * lp(np.abs(r.standard_normal(len(tt))), 12) / 0.1
    x = saw * am * adsr(len(tt), 0.3, 0.6, 0.8, 1.2)
    x = bp(x, 60, 900) + bp(pink(sec, r), 200, 700) * 0.4 * adsr(len(tt), 0.3, 0.6, 0.8, 1.2)
    x = fx(x.astype(np.float32), Distortion(drive_db=12), LowpassFilter(cutoff_frequency_hz=1400))
    return normalize(reverb(x, 0.8, 0.35, 0.85))


@sfx
def sfx_tyrant_footstep():
    x = boom(1.8, seed=38, fc=250, sub=40)
    return normalize(lp(x, 400))


@sfx
def sfx_bone_snap():
    x = noise_burst(0.05, 0.006, lo=800, hi=6000, seed=39)
    x = np.concatenate([x, np.zeros(int(0.02 * SR), np.float32), noise_burst(0.04, 0.005, lo=1000, hi=5000, seed=40) * 0.6])
    return normalize(reverb(x, 0.3, 0.2, 0.9), -3)


@sfx
def sfx_body_fall():
    x = lp(mixp(thump(65, 0.5, 0.07, 40), noise_burst(0.2, 0.03, hi=1500, seed=41) * 0.5), 1800)
    return normalize(reverb(x, 0.4, 0.2, 0.9), -3)


@sfx
def sfx_punch():
    x = lp(mixp(thump(110, 0.2, 0.03, 60), noise_burst(0.06, 0.008, hi=4000, seed=42) * 0.7), 5000)
    return normalize(reverb(x, 0.3, 0.15, 0.95), -2)


@sfx
def sfx_metal_clang():
    parts = [(310, 1.0), (547, 0.6), (873, 0.5), (1290, 0.35), (1811, 0.25), (2533, 0.15)]
    x = sum(fm_bell(f, 2.5, index=0.2, ratio=1.0, tau=0.9) * a for f, a in parts)
    x[:int(0.02 * SR)] += noise_burst(0.02, 0.004, lo=1000, seed=43)
    return normalize(reverb(x, 0.7, 0.35, 0.8))


@sfx
def sfx_laser():
    sec = 0.6
    x = sine_sweep(sec, 2400, 180) * expdecay(sec, 0.15)
    x = x + 0.4 * np.sign(x) * expdecay(sec, 0.1)
    return normalize(reverb(lp(x, 7000), 0.6, 0.3, 0.8), -3)


@sfx
def sfx_ship_pass():
    sec = 7.0
    n = int(sec * SR)
    tt = t(sec)
    env = np.exp(-((tt - 3.2) / 1.3) ** 2)
    rum = lp(brown(sec, rng(44)), 300) * env
    tone = (np.sin(2 * np.pi * 55 * tt) + 0.5 * np.sin(2 * np.pi * 82.5 * tt)) * env * 0.4
    hiss = bp(white(sec, rng(45)), 1500, 6000) * env * 0.1
    x = rum + tone + hiss
    p = np.linspace(-0.9, 0.9, n)
    y = np.stack([x * np.cos((p + 1) * np.pi / 4), x * np.sin((p + 1) * np.pi / 4)])
    return normalize(reverb(y, 0.9, 0.4, 0.8))


@sfx
def sfx_alien_call():
    sec = 4.0
    tt = t(sec)
    f = 180 + 60 * np.sin(2 * np.pi * 0.6 * tt) + 20 * np.sin(2 * np.pi * 5 * tt)
    ph = 2 * np.pi * np.cumsum(f) / SR
    v = sum(np.sin(k * ph) * (1 / k) for k in range(1, 12))
    ring = v * np.sin(2 * np.pi * 37 * tt)
    x = bp(v * 0.6 + ring * 0.4, 150, 3000) * adsr(len(tt), 0.5, 0.5, 0.8, 1.5)
    x = fx(x.astype(np.float32), Chorus(rate_hz=0.4, depth=0.6, mix=0.6))
    return normalize(reverb(x, 0.95, 0.5, 0.6))


@sfx
def sfx_radio_static():
    sec = 1.5
    x = bp(white(sec, rng(46)), 800, 5000) * adsr(int(sec * SR), 0.01, 0.2, 0.6, 0.4) * 0.5
    beep = np.sin(2 * np.pi * 1200 * t(0.12)) * 0.4
    x[-len(beep):] += beep
    return normalize(x, -6)


@sfx
def sfx_heart_monitor():
    sec = 5.0
    buf = np.zeros(int(sec * SR), np.float32)
    for k in range(5):
        b = np.sin(2 * np.pi * 1000 * t(0.12)) * adsr(int(0.12 * SR), 0.005, 0.02, 0.7, 0.05)
        i = int(k * 1.0 * SR)
        buf[i:i + len(b)] += b * 0.5
    return normalize(reverb(buf, 0.4, 0.2, 0.9), -6)


@sfx
def sfx_caption():
    """Soft data blip under a typed location caption."""
    b = mixp(fm_bell(1567.98, 0.5, index=0.5, ratio=3.0, tau=0.12) * 0.4, fm_bell(2093, 0.4, index=0.3, tau=0.08) * 0.2)[0]
    return normalize(reverb(b, 0.5, 0.25, 0.8), -9)


@sfx
def sfx_title_hit():
    x = boom(5.0, seed=47, fc=500, sub=42)
    return normalize(x)


@sfx
def sfx_paper():
    sec = 0.8
    x = bp(white(sec, rng(48)), 1500, 8000) * adsr(int(sec * SR), 0.05, 0.2, 0.3, 0.4)
    x *= (0.6 + 0.4 * (rng(49).random(len(x)) > 0.6))
    return normalize(reverb(lp(x, 9000), 0.3, 0.15, 0.9), -8)


@sfx
def sfx_pen_sign():
    sec = 1.6
    tt = t(sec)
    stroke = (np.sin(2 * np.pi * 3 * tt) > -0.2).astype(np.float32)
    x = bp(white(sec, rng(50)), 2500, 7000) * stroke * adsr(len(tt), 0.05, 0.2, 0.8, 0.2) * 0.4
    return normalize(x, -10)


@sfx
def sfx_bell_stock():
    """An exchange-style opening bell."""
    sec = 5.0
    buf = np.zeros(int(sec * SR), np.float32)
    for k in range(10):
        b = sum(fm_bell(f, 1.5, index=0.2, tau=0.7) * a for f, a in [(880, 0.5), (1760, 0.25), (2640, 0.1)])
        i = int(k * 0.28 * SR)
        buf[i:i + len(b)] += b[:len(buf) - i]
    return normalize(reverb(buf, 0.7, 0.35, 0.8), -3)


@sfx
def sfx_gunshot_suppressed():
    r = rng(60)
    thwack = noise_burst(0.06, 0.012, lo=600, hi=4500, seed=60)
    click = noise_burst(0.008, 0.0015, lo=3000, seed=61) * 0.8
    mech = noise_burst(0.03, 0.006, lo=1500, hi=6000, seed=62) * 0.5
    x = np.zeros(int(0.4 * SR), np.float32)
    x[:len(click)] += click
    x[:len(thwack)] += thwack
    x[int(0.05 * SR):int(0.05 * SR) + len(mech)] += mech
    return normalize(reverb(x, 0.35, 0.2, 0.9), -2)


@sfx
def sfx_suppressed_burst():
    buf = np.zeros((2, int(1.6 * SR)), np.float32)
    shot = sfx_gunshot_suppressed()
    r = rng(63)
    for k in range(6):
        place(buf, shot, k * r.uniform(0.09, 0.12), r.uniform(0.7, 1.0))
    return normalize(buf, -1.5)


@sfx
def sfx_alarm():
    sec = 6.0
    tt = t(sec)
    gate = ((tt % 0.5) < 0.3).astype(np.float32)
    x = (np.sign(np.sin(2 * np.pi * 880 * tt)) * 0.3 + np.sin(2 * np.pi * 880 * tt) * 0.4) * gate
    x = bp(x, 500, 4000) * adsr(len(tt), 0.05, 0.1, 0.9, 1.0)
    return normalize(reverb(x, 0.7, 0.35, 0.7), -4)


@sfx
def sfx_radio_chatter():
    sec = 2.5
    x = bp(white(sec, rng(64)), 700, 3500) * 0.15
    beep = np.sin(2 * np.pi * 1400 * t(0.08)) * 0.5
    x[:len(beep)] += beep
    x[-len(beep):] += beep
    return normalize(x * adsr(int(sec * SR), 0.01, 0.1, 0.8, 0.1), -8)


@sfx
def sfx_police_sirens():
    a = sfx_siren()
    b = np.roll(sfx_siren(), int(1.3 * SR), axis=-1) * 0.7
    c = mixp(a, b)
    return normalize(reverb(c, 0.9, 0.4, 0.7), -3)


@sfx
def sfx_car_crash():
    x = boom(3.0, seed=65, fc=2500, sub=60)
    g = sfx_glass_break()
    m = sfx_metal_clang()
    return normalize(mixp(x, g * 0.6, m * 0.5))


def build(names=None, force=False):
    os.makedirs(OUT, exist_ok=True)
    names = names or list(REG)
    for name in names:
        path = os.path.join(OUT, f'{name}.wav')
        if os.path.exists(path) and not force:
            continue
        x = REG[name]()
        if x.ndim == 1:
            x = np.stack([x, x])
        x = np.nan_to_num(x)
        if not name.startswith('amb_'):
            x = fade(x, 0.002, 0.08)
        save(path, x)
        print(f'{name:26s} {x.shape[1] / SR:6.2f}s  peak {20 * np.log10(np.abs(x).max() + 1e-9):6.1f} dB  rms {rms_db(x):6.1f} dB', flush=True)


if __name__ == '__main__':
    build(sys.argv[1:] or None, force='--force' in sys.argv)
