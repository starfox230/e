"""Crowd sounds built from many layered Kokoro voices: murmur, cheer, chant, gasp, reverent silence.

Run: nice python3 pipeline/crowd_lib.py [name ...] [--force]
"""
import os
import sys
import numpy as np
from scipy import signal
from pedalboard import PitchShift
from dsp import SR, save, reverb, normalize, lp, hp, bp, pink, place, pan, fade, rms_db, t, adsr, make_loop, fx
from sfx_lib import OUT, sfx_applause

VOICES = ['am_michael', 'am_fenrir', 'am_puck', 'am_echo', 'am_eric', 'am_liam', 'am_onyx', 'am_adam',
          'af_heart', 'af_bella', 'af_nicole', 'af_sarah', 'af_sky', 'af_river', 'af_nova', 'af_kore', 'af_aoede',
          'bm_george', 'bm_lewis', 'bm_daniel', 'bm_fable', 'bf_emma', 'bf_isabella', 'bf_alice', 'bf_lily']
RNG = np.random.default_rng(2035)
_k = None
REG = {}


def crowd(fn):
    REG[fn.__name__] = fn
    return fn


def kokoro():
    global _k
    if _k is None:
        from kokoro_onnx import Kokoro
        _k = Kokoro('/opt/models/kokoro/kokoro-v1.0.onnx', '/opt/models/kokoro/voices-v1.0.bin')
    return _k


def say(text, voice, speed=1.0):
    a, sr = kokoro().create(text, voice=voice, speed=speed, lang='en-gb' if voice[0] == 'b' else 'en-us')
    a = signal.resample_poly(a.astype(np.float32), SR // sr, 1)
    idx = np.where(np.abs(a) > 0.01)[0]
    return a[idx[0]:idx[-1] + 1] if len(idx) else a


def shift(x, st):
    return fx(x, PitchShift(semitones=st))[0] if st else x


def stadium(x, wet=0.45):
    return reverb(x, room=0.97, wet=wet, dry=0.55, damp=0.35, width=1.0)


def roar(sec, seed=0):
    """Vowel-shaped noise: the 'aaah' of thousands."""
    r = np.random.default_rng(seed)
    n = pink(sec, r)
    a = bp(n, 550, 900) * 0.8 + bp(n, 1000, 1500) * 0.5 + bp(n, 2300, 2900) * 0.2
    return a.astype(np.float32)


@crowd
def amb_crowd_murmur():
    lines = ['I heard the new building is incredible', 'did you see the numbers this quarter', 'where are we sitting',
             'my daughter wanted to come so badly', 'is he actually going to speak', 'I think it starts at eight',
             'we flew in from Singapore yesterday', 'have you been to Basel', 'this is so much bigger than last year',
             'they said there would be drones', 'I can not believe how many people', 'I need a coffee',
             'the kids have been practising the creed all week', 'look at the screens', 'honestly I am nervous']
    sec = 50
    buf = np.zeros((2, int(sec * SR)), np.float32)
    for k in range(70):
        v = VOICES[k % len(VOICES)]
        x = say(lines[k % len(lines)], v, speed=RNG.uniform(0.95, 1.1))
        x = lp(x, 3500) * RNG.uniform(0.3, 1.0)
        place(buf, pan(x, RNG.uniform(-1, 1)), RNG.uniform(0, sec - 4))
    buf += np.stack([roar(sec, 1), roar(sec, 2)]) * 0.08
    buf = stadium(lp(buf, 5000), 0.5)[:, :int(sec * SR)]
    return make_loop(buf * 10 ** ((-22 - rms_db(buf)) / 20), 4)


@crowd
def sfx_crowd_cheer():
    sec = 9.0
    buf = np.zeros((2, int(sec * SR)), np.float32)
    env = adsr(int(sec * SR), 0.6, 1.5, 0.75, 4.0)
    words = ['Yeah!', 'Whoo!', 'Yes!', 'Umbrella!', 'Wesker!', 'Hey!']
    for k in range(90):
        v = VOICES[k % len(VOICES)]
        x = say(words[k % len(words)], v, speed=RNG.uniform(1.0, 1.2))
        x = shift(x, RNG.choice([-2, -1, 0, 1, 2]))
        at = RNG.uniform(0.2, 5.5)
        place(buf, pan(x * RNG.uniform(0.3, 1.0), RNG.uniform(-1, 1)), at)
    r = np.stack([roar(sec, 3), roar(sec, 4)]) * 1.2
    buf = buf * 0.8 + r
    buf *= env
    ap = sfx_applause()
    place(buf, ap, 0.3, 0.6)
    return normalize(stadium(buf, 0.4), -1)


@crowd
def sfx_crowd_gasp():
    sec = 3.0
    buf = np.zeros((2, int(sec * SR)), np.float32)
    for k in range(40):
        v = VOICES[k % len(VOICES)]
        x = say(RNG.choice(['Oh!', 'Ah!', 'Oh my god']), v, speed=1.1)
        place(buf, pan(x * RNG.uniform(0.3, 0.9), RNG.uniform(-1, 1)), RNG.uniform(0.0, 0.35))
    return normalize(stadium(lp(buf, 6000), 0.4), -3)


def chant(word='Umbrella!', reps=6, beat=1.55, voices=64, sec=None):
    sec = sec or reps * beat + 4
    buf = np.zeros((2, int(sec * SR)), np.float32)
    takes = []
    for k in range(voices):
        v = VOICES[k % len(VOICES)]
        x = say(word, v, speed=RNG.uniform(1.05, 1.2))
        takes.append((shift(x, RNG.choice([-3, -2, -1, 0, 1])), RNG.uniform(-1, 1), RNG.uniform(0.35, 1.0)))
    for rep in range(reps):
        swell = 0.6 + 0.4 * rep / max(1, reps - 1)
        for x, p, g in takes:
            place(buf, pan(x * g * swell, p), 0.3 + rep * beat + RNG.normal(0, 0.035))
        # stomps and claps on the beat
        for off in (0.0, beat * 0.5):
            thump = lp(RNG.standard_normal(int(0.12 * SR)).astype(np.float32) * np.exp(-np.linspace(0, 12, int(0.12 * SR))), 160)
            place(buf, np.stack([thump, thump]) * 1.5 * swell, 0.3 + rep * beat + off + beat * 0.62)
    buf += np.stack([roar(sec, 5), roar(sec, 6)]) * 0.25 * adsr(int(sec * SR), 0.5, 1, 0.8, 2)
    return normalize(stadium(buf, 0.42), -1)


@crowd
def sfx_chant_umbrella():
    return chant('Umbrella!', reps=6)


@crowd
def sfx_chant_umbrella_long():
    return chant('Umbrella!', reps=12, voices=80)


@crowd
def sfx_creed_crowd():
    """Many voices saying 'We are Umbrella' together, like the end of a pledge."""
    return chant('We are Umbrella.', reps=1, beat=2.5, voices=90, sec=7)


@crowd
def sfx_crowd_we_are_enough():
    return chant('We are enough.', reps=1, beat=2.5, voices=90, sec=7)


def build(names=None, force=False):
    os.makedirs(OUT, exist_ok=True)
    for name in names or list(REG):
        path = os.path.join(OUT, f'{name}.wav')
        if os.path.exists(path) and not force:
            continue
        x = REG[name]()
        x = fade(x, 0.01, 0.2) if not name.startswith('amb_') else x
        save(path, x)
        print(f'{name:26s} {x.shape[1] / SR:6.2f}s  rms {rms_db(x):6.1f} dB', flush=True)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    build(args or None, force='--force' in sys.argv)
