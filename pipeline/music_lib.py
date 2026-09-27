"""Score: MIDI cues composed in code, rendered with FluidSynth + MuseScore General, mastered with pedalboard.

The Umbrella motif is a descending four-note figure: 6 - 5 - 3 - b7 (in G minor: Eb D Bb F).
Run `python3 pipeline/music_lib.py [cue ...] [--force]`.
"""
import os
import sys
import subprocess
import numpy as np
import pretty_midi as pm
from dsp import SR, load, save, reverb, normalize, fx, rms_db, fade, lp, hp
from pedalboard import Compressor, Limiter, HighShelfFilter, LowShelfFilter, Gain

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'assets', 'music')
WORK = os.path.join(ROOT, 'work', 'music')
SF = '/usr/share/sounds/sf3/MuseScore_General.sf3'
RNG = np.random.default_rng(1933)
REG = {}

NOTE = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7,
        'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}


def n(name):
    """'D4' -> midi number."""
    pc = name[:-1]
    octv = int(name[-1])
    return 12 * (octv + 1) + NOTE[pc]


def chord(root, quality='m', octave=3):
    r = n(f'{root}{octave}')
    ints = {'m': [0, 3, 7], 'M': [0, 4, 7], 'dim': [0, 3, 6], 'sus4': [0, 5, 7], 'sus2': [0, 2, 7],
            'm7': [0, 3, 7, 10], 'M7': [0, 4, 7, 11], '7': [0, 4, 7, 10], 'm9': [0, 3, 7, 10, 14],
            'add9': [0, 4, 7, 14], 'madd9': [0, 3, 7, 14], '5': [0, 7, 12]}[quality]
    return [r + i for i in ints]


class Cue:
    def __init__(self, bpm=60, seed=0):
        self.m = pm.PrettyMIDI(initial_tempo=bpm)
        self.bpm = bpm
        self.b = 60.0 / bpm
        self.r = np.random.default_rng(seed)
        self.inst = {}
        self.length = 0.0

    def track(self, key, program, drum=False):
        if key not in self.inst:
            i = pm.Instrument(program=program, is_drum=drum, name=key)
            self.m.instruments.append(i)
            self.inst[key] = i
        return self.inst[key]

    def note(self, key, program, pitch, beat, dur, vel=70, human=True, drum=False):
        tr = self.track(key, program, drum)
        s = beat * self.b + (self.r.normal(0, 0.008) if human else 0)
        e = s + dur * self.b
        v = int(np.clip(vel + (self.r.normal(0, 5) if human else 0), 1, 127))
        tr.notes.append(pm.Note(velocity=v, pitch=int(pitch), start=max(0, s), end=max(s + 0.02, e)))
        self.length = max(self.length, e)

    def notes(self, key, program, pitches, beat, dur, vel=70, **kw):
        for p in pitches:
            self.note(key, program, p, beat, dur, vel, **kw)

    def swell(self, key, program, beat0, beat1, v0, v1, cc=11, shape='sin'):
        tr = self.track(key, program)
        steps = max(2, int((beat1 - beat0) * self.b / 0.1))
        for k in range(steps + 1):
            f = k / steps
            if shape == 'sin':
                f = 0.5 - 0.5 * np.cos(np.pi * f)
            val = int(v0 + (v1 - v0) * f)
            tr.control_changes.append(pm.ControlChange(cc, int(np.clip(val, 0, 127)), (beat0 + (beat1 - beat0) * k / steps) * self.b))

    def cc(self, key, program, num, val, beat):
        tr = self.track(key, program)
        tr.control_changes.append(pm.ControlChange(num, int(val), beat * self.b))

    def bend(self, key, program, cents, beat):
        tr = self.track(key, program)
        tr.pitch_bends.append(pm.PitchBend(int(np.clip(cents / 200 * 8191, -8192, 8191)), beat * self.b))


def cue(fn):
    REG[fn.__name__] = fn
    return fn


def render(name, c: Cue, tail=6.0, gain=0.55, room=0.85, wet=0.28, target_rms=-20.0, loop=True):
    os.makedirs(WORK, exist_ok=True)
    mid = os.path.join(WORK, f'{name}.mid')
    raw = os.path.join(WORK, f'{name}.raw.wav')
    # every channel gets reverb/chorus send 0 (we do our own reverb) and full volume
    for inst in c.m.instruments:
        inst.control_changes.insert(0, pm.ControlChange(91, 0, 0))
        inst.control_changes.insert(0, pm.ControlChange(93, 0, 0))
        inst.control_changes.insert(0, pm.ControlChange(7, 110, 0))
    c.m.write(mid)
    subprocess.run(['fluidsynth', '-ni', '-q', '-g', str(gain), '-r', str(SR), '-F', raw, SF, mid], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    x = load(raw)
    x = x[:, :int((c.length + tail) * SR)]
    x = reverb(x, room=room, wet=wet, dry=0.85, damp=0.45, width=1.0)
    x = fx(x, LowShelfFilter(cutoff_frequency_hz=120, gain_db=1.5), HighShelfFilter(cutoff_frequency_hz=9000, gain_db=-2.0),
           Compressor(threshold_db=-20, ratio=2.0, attack_ms=30, release_ms=300))
    x = x * 10 ** ((target_rms - rms_db(x)) / 20)
    x = fx(x, Limiter(threshold_db=-2.0))
    x = fade(x, 0.05, 3.0 if not loop else 0.05)
    save(os.path.join(OUT, f'{name}.wav'), x)
    os.remove(raw)
    return x


def motif(c, key, program, beat, octave=5, vel=62, root='G', dur=1.0):
    """The Umbrella motif in the given minor key root."""
    base = n(f'{root}{octave}')
    for k, iv in enumerate([8, 7, 3, -2]):  # b6 5 b3 b7(below)
        c.note(key, program, base + iv, beat + k * dur, dur * (2.0 if k == 3 else 0.95), vel - k * 3)


# ------------------------------------------------------------------ cues

@cue
def m_bunker():
    c = Cue(56, 1)
    bars = 36
    prog = [('D', 'm'), ('Bb', 'M'), ('G', 'm'), ('A', 'M')]
    for bar in range(bars):
        b = bar * 4
        root, q = prog[(bar // 2) % 4]
        if bar % 2 == 0:
            ch = chord(root, q, 2)
            c.notes('bass', 43, [ch[0] - 12, ch[0]], b, 8.1, 60)
            c.notes('cello', 42, [ch[0], ch[2]], b, 8.1, 52)
            if bar >= 8:
                c.notes('trom', 57, chord(root, q, 3), b + 0.5, 7.5, 40)
            if bar >= 16:
                c.notes('choir', 52, [ch[0] + 12, ch[2] + 12], b, 8.0, 38)
        if bar % 8 == 6:
            for k in range(32):  # timpani roll swelling
                c.note('timp', 47, n('D2'), b + k * 0.25, 0.25, 25 + k * 2)
            c.note('timp', 47, n('D2'), b + 8, 2, 90)
    c.swell('bass', 43, 0, bars * 4, 60, 110)
    c.swell('cello', 42, 0, bars * 4, 50, 105)
    c.swell('trom', 57, 32, bars * 4, 30, 90)
    return render('m_bunker', c, room=0.92, wet=0.35, target_rms=-21)


@cue
def m_horror():
    c = Cue(50, 2)
    bars = 26
    for bar in range(bars):
        b = bar * 4
        if bar % 2 == 0:
            c.notes('trem_lo', 44, [n('D2'), n('Eb2')], b, 8.2, 55)
            c.notes('trem_hi', 44, [n('A4'), n('Bb4'), n('D5')] if bar % 4 == 0 else [n('Ab4'), n('A4'), n('Eb5')], b, 8.2, 40)
        if bar % 3 == 1:
            c.notes('piano', 0, [n('D1'), n('Eb1'), n('Ab1'), n('D2')], b + 1.3, 6, 80)
        if bar % 4 == 2:
            c.notes('choir', 52, [n('D4'), n('Eb4')], b, 12, 40)
        if bar % 6 == 5:
            c.note('vln', 40, n('A6'), b, 10, 35)
    c.swell('trem_lo', 44, 0, bars * 4, 40, 95)
    c.swell('trem_hi', 44, 0, bars * 4, 20, 80)
    c.swell('choir', 52, 0, bars * 4, 40, 90)
    return render('m_horror', c, room=0.95, wet=0.4, target_rms=-22)


@cue
def m_rain_piano():
    c = Cue(66, 3)
    prog = [('G', 'm'), ('Eb', 'M'), ('C', 'm'), ('D', 'sus4')]
    bars = 40
    c.bend('piano', 0, -14, 0)  # the tired break-room piano, slightly flat
    for bar in range(bars):
        b = bar * 4
        root, q = prog[bar % 4]
        pad = chord(root, q, 3)
        c.notes('pad', 89, pad, b, 4.1, 45)
        c.note('pbass', 0, pad[0] - 12, b, 3.8, 48)
        if bar % 4 == 0:
            motif(c, 'piano', 0, b, octave=5 if (bar // 4) % 2 == 0 else 4, vel=64)
        else:
            for k, p in enumerate(sorted(pad)[1:]):
                if c.r.random() < 0.55:
                    c.note('piano', 0, p + 12, b + 1 + k * 1.0 + c.r.uniform(0, 0.2), 1.6, 42)
        if bar >= 12:
            c.notes('str', 49, [pad[0] + 12, pad[1] + 12], b, 4.1, 44)
        if bar % 4 == 3:
            c.note('pad', 89, pad[0] + 12, b, 4, 40)
    c.swell('str', 49, 48, bars * 4, 40, 95)
    return render('m_rain_piano', c, room=0.85, wet=0.32, target_rms=-22)


@cue
def m_system():
    c = Cue(80, 4)
    prog = [('E', 'm9'), ('C', 'M7'), ('A', 'm9'), ('B', 'sus4')]
    bars = 32
    for bar in range(bars):
        b = bar * 4
        root, q = prog[(bar // 2) % 4]
        ch = chord(root, q, 3)
        if bar % 2 == 0:
            c.notes('pad', 88, ch, b, 8.1, 50)
            c.notes('sweep', 95, [ch[0] + 12, ch[2] + 12], b, 8.1, 40)
            c.note('sub', 38, ch[0] - 12, b, 7.5, 55)
        arp = sorted(set([p + 24 for p in ch]))
        for k in range(8):
            c.note('cel', 8, arp[k % len(arp)], b + k * 0.5, 0.45, 38 + (k % 4) * 4)
    c.swell('pad', 88, 0, bars * 4, 70, 100)
    return render('m_system', c, room=0.9, wet=0.35, target_rms=-23)


@cue
def m_dawn():
    c = Cue(58, 5)
    prog = [('D', 'm'), ('Bb', 'M'), ('F', 'M'), ('C', 'M'), ('D', 'm'), ('Bb', 'M'), ('G', 'm'), ('A', 'M'),
            ('Bb', 'M'), ('F', 'M'), ('C', 'M'), ('D', 'm'), ('Bb', 'M'), ('C', 'M'), ('F', 'M'), ('F', 'M')]
    melody = ['A4', 'F4', 'G4', 'E4', 'F4', 'D4', 'D4', 'C#4', 'D5', 'C5', 'E5', 'F5', 'D5', 'E5', 'C5', 'A4']
    for k, (root, q) in enumerate(prog):
        b = k * 4
        ch = chord(root, q, 3)
        c.notes('str', 48, [ch[0] - 12] + ch, b, 4.1, 58)
        c.note('bass', 43, ch[0] - 24, b, 4.1, 55)
        if k >= 4:
            c.note('horn', 60, n(melody[k]), b, 3.8, 62)
        if k >= 8:
            c.notes('vln', 49, [p + 12 for p in ch], b, 4.1, 50)
            c.note('choir', 52, ch[0] + 12, b, 4.1, 45)
    c.note('timp', 47, n('F2'), 56, 1, 70)
    for k in range(16):
        c.note('timp', 47, n('F2'), 52 + k * 0.25, 0.25, 30 + k * 3)
    c.swell('str', 48, 0, 64, 55, 110)
    c.swell('vln', 49, 32, 64, 40, 105)
    return render('m_dawn', c, room=0.9, wet=0.33, target_rms=-20, loop=False)


@cue
def m_tension():
    c = Cue(100, 6)
    bars = 40
    pattern = ['D2', 'D2', 'D3', 'D2', 'F2', 'D2', 'Eb2', 'D2']
    for bar in range(bars):
        b = bar * 4
        shift = [0, 0, -2, 1][(bar // 4) % 4]
        for k, p in enumerate(pattern):
            c.note('ost', 48, n(p) + shift, b + k * 0.5, 0.3, 64 + (12 if k == 0 else 0))
        if bar % 2 == 0:
            c.note('lopiano', 0, n('D1') + shift, b, 3, 70)
        if bar >= 8:
            c.note('vln', 44, n('A5') + (1 if bar % 8 >= 4 else 0), b, 4.05, 40)
        if bar >= 16 and bar % 4 == 0:
            c.note('kick', 0, 36, b, 0.5, 60, drum=True)
            c.note('kick', 0, 36, b + 2.5, 0.5, 45, drum=True)
    c.swell('ost', 48, 0, bars * 4, 60, 105)
    return render('m_tension', c, room=0.7, wet=0.22, target_rms=-21)


@cue
def m_scheme():
    c = Cue(92, 7)
    bars = 40
    prog = [('A', 'm'), ('F', 'M'), ('D', 'm'), ('E', 'sus4')]
    riff = [0, 7, 12, 7, 3, 7, 12, 15]
    for bar in range(bars):
        b = bar * 4
        root, q = prog[(bar // 2) % 4]
        ch = chord(root, q, 3)
        for k, iv in enumerate(riff):
            c.note('pizz', 45, ch[0] + iv, b + k * 0.5, 0.4, 60 + (10 if k % 4 == 0 else 0))
        if bar % 2 == 0:
            c.notes('pad', 89, ch, b, 8.1, 42)
            c.note('bass', 32, ch[0] - 12, b, 7.5, 60)
        if bar >= 8:
            for k in range(4):
                c.note('stick', 0, 37, b + k + 0.5, 0.2, 40, drum=True)
        if bar >= 16 and bar % 4 == 0:
            c.note('cel', 8, ch[2] + 24, b, 2, 45)
            c.note('cel', 8, ch[1] + 24, b + 2, 2, 42)
    return render('m_scheme', c, room=0.6, wet=0.2, target_rms=-22)


@cue
def m_sermon():
    c = Cue(66, 8)
    prog = [('C', 'm'), ('Ab', 'M'), ('Eb', 'M'), ('Bb', 'M'), ('F', 'm'), ('Ab', 'M'), ('G', 'M'), ('G', 'M')]
    loops = 4
    for L in range(loops):
        for k, (root, q) in enumerate(prog):
            b = (L * 8 + k) * 4
            ch = chord(root, q, 3)
            c.notes('str', 48, [ch[0] - 12] + ch, b, 4.1, 50 + L * 10)
            c.note('bass', 43, ch[0] - 24, b, 4.1, 55 + L * 10)
            if L >= 1:
                c.notes('choir', 52, [p + 12 for p in ch], b, 4.1, 45 + L * 12)
            if L >= 2:
                c.notes('brass', 61, ch, b, 3.9, 50 + L * 12)
                c.note('timp', 47, ch[0] - 12 if ch[0] - 12 >= n('F2') else ch[0], b, 1, 60 + L * 10)
            if L >= 3:
                c.note('tpt', 56, ch[2] + 12, b, 3.8, 80)
                if k == 7:
                    c.note('cym', 0, 49, b + 4, 2, 90, drum=True)
    c.swell('str', 48, 0, loops * 32, 55, 115)
    c.swell('choir', 52, 32, loops * 32, 50, 115)
    return render('m_sermon', c, room=0.93, wet=0.36, target_rms=-19, loop=False)


@cue
def m_march():
    c = Cue(112, 9)
    bars = 48
    tune = ['D4', 'D4', 'A4', 'A4', 'Bb4', 'A4', 'G4', 'F4', 'E4', 'F4', 'G4', 'A4', 'D4', 'E4', 'F4', 'D4']
    prog = [('D', 'm'), ('D', 'm'), ('Bb', 'M'), ('A', 'M')]
    for bar in range(bars):
        b = bar * 4
        root, q = prog[bar % 4]
        ch = chord(root, q, 3)
        # snare pattern
        for k, v in enumerate([90, 0, 60, 55, 85, 0, 60, 70]):
            if v:
                c.note('sn', 0, 38, b + k * 0.5, 0.2, v, drum=True)
        c.note('bd', 0, 36, b, 0.3, 95, drum=True)
        c.note('bd', 0, 36, b + 2, 0.3, 80, drum=True)
        c.note('tuba', 58, ch[0] - 12, b, 0.9, 80)
        c.note('tuba', 58, ch[2] - 12, b + 2, 0.9, 72)
        c.notes('horn', 60, ch, b + 1, 0.5, 60)
        c.notes('horn', 60, ch, b + 3, 0.5, 60)
        if bar >= 8:
            m1, m2 = tune[(bar * 2) % 16], tune[(bar * 2 + 1) % 16]
            c.note('tpt', 56, n(m1), b, 1.9, 85)
            c.note('tpt', 56, n(m2), b + 2, 1.9, 82)
            c.note('trom', 57, n(m1) - 12, b, 1.9, 70)
        if bar % 8 == 7:
            c.note('cym', 0, 49, b + 4, 2, 90, drum=True)
    return render('m_march', c, room=0.8, wet=0.25, target_rms=-19)


@cue
def m_action():
    c = Cue(140, 10)
    bars = 48
    for bar in range(bars):
        b = bar * 4
        shift = [0, 0, 3, -2][(bar // 4) % 4]
        for k in range(16):
            c.note('ost', 48, n('D3') + shift + (12 if k % 4 == 2 else 0), b + k * 0.25, 0.2, 70 + (15 if k % 4 == 0 else 0))
        for k, v in enumerate([100, 0, 0, 80, 0, 0, 90, 0]):
            if v:
                c.note('taiko', 116, n('C3'), b + k * 0.5, 0.4, v)
        c.note('tom', 0, 45, b + 3.5, 0.2, 80, drum=True)
        c.note('tom', 0, 43, b + 3.75, 0.2, 85, drum=True)
        if bar % 2 == 0:
            c.notes('brass', 61, [n('D3') + shift, n('A3') + shift], b, 1.5, 95)
            c.note('lobrass', 58, n('D2') + shift, b, 1.5, 95)
        if bar % 8 == 7:
            c.note('cym', 0, 49, b + 4, 2, 100, drum=True)
    return render('m_action', c, room=0.75, wet=0.22, target_rms=-18)


@cue
def m_triumph():
    c = Cue(64, 11)
    prog = [('D', 'm'), ('Bb', 'M'), ('C', 'M'), ('A', 'm'), ('Bb', 'M'), ('G', 'm'), ('A', 'sus4'), ('A', 'M')]
    loops = 3
    for L in range(loops):
        for k, (root, q) in enumerate(prog):
            b = (L * 8 + k) * 4
            ch = chord(root, q, 3)
            c.notes('str', 48, [ch[0] - 12] + ch + [ch[1] + 12], b, 4.1, 70 + L * 10)
            c.notes('brass', 61, ch, b, 3.9, 60 + L * 12)
            c.notes('choir', 52, [p + 12 for p in ch], b, 4.1, 55 + L * 12)
            c.note('bass', 43, ch[0] - 24, b, 4.1, 80)
            c.note('timp', 47, n('D2') if root in 'DA' else n('F2'), b, 1, 80)
            c.note('tpt', 56, ch[2] + 12, b + 1, 2.8, 70 + L * 10)
            if k % 4 == 3:
                c.note('cym', 0, 49, b + 4, 2, 80 + L * 10, drum=True)
    return render('m_triumph', c, room=0.93, wet=0.35, target_rms=-18, loop=False)


@cue
def m_sad():
    c = Cue(58, 12)
    prog = [('F', 'm'), ('Db', 'M'), ('Ab', 'M'), ('Eb', 'M'), ('F', 'm'), ('Db', 'M'), ('Bb', 'm'), ('C', 'M')]
    mel = ['C5', 'Ab4', 'Eb5', 'Bb4', 'C5', 'F4', 'Db5', 'E4']
    for L in range(3):
        for k, (root, q) in enumerate(prog):
            b = (L * 8 + k) * 4
            ch = chord(root, q, 3)
            for j, p in enumerate([ch[0] - 12, ch[2] - 12, ch[1], ch[2]]):
                c.note('piano', 0, p, b + j, 1.8, 45)
            if L >= 1:
                c.note('cello', 42, n(mel[k]) - 12, b, 3.9, 60)
            if L >= 2:
                c.notes('str', 49, ch, b, 4.1, 45)
    return render('m_sad', c, room=0.9, wet=0.33, target_rms=-22)


@cue
def m_space():
    c = Cue(48, 13)
    prog = [('D', 'M'), ('E', 'M'), ('B', 'm'), ('G', 'M')]
    bars = 32
    for bar in range(0, bars, 2):
        b = bar * 4
        root, q = prog[(bar // 2) % 4]
        ch = chord(root, q, 3)
        c.notes('halo', 94, ch + [ch[0] + 12], b, 8.2, 50)
        c.notes('cpad', 91, [ch[0] + 12, ch[2] + 12], b, 8.2, 40)
        c.note('sub', 38, ch[0] - 24, b, 8.0, 50)
        for k in range(4):
            if c.r.random() < 0.7:
                c.note('cel', 8, ch[c.r.integers(3)] + 36, b + k * 2 + c.r.uniform(0, 1), 2, 30)
    return render('m_space', c, room=0.97, wet=0.45, target_rms=-23)


@cue
def m_alien():
    c = Cue(45, 14)
    bars = 28
    wt = [n('C3'), n('D3'), n('E3'), n('F#3'), n('G#3'), n('A#3')]
    for bar in range(0, bars, 2):
        b = bar * 4
        sel = [wt[(bar // 2 + k * 2) % 6] for k in range(3)]
        c.notes('metal', 93, sel, b, 8.2, 50)
        c.notes('bowed', 92, [p + 12 for p in sel], b + 1, 7, 40)
        c.note('choir', 52, sel[0] + 12, b, 8.2, 35)
        c.bend('choir', 52, c.r.uniform(-60, 60), b + 2)
        c.bend('choir', 52, 0, b + 6)
    return render('m_alien', c, room=0.97, wet=0.45, target_rms=-23)


@cue
def m_title():
    c = Cue(60, 15)
    c.notes('hit', 55, [n('G2'), n('D3'), n('G3')], 0, 1.5, 100)
    c.note('timp', 47, n('G2'), 0, 2, 110)
    c.notes('lostr', 48, [n('G1'), n('G2'), n('D3')], 0, 6, 70)
    motif(c, 'piano', 0, 1.2, octave=5, vel=70, dur=0.8)
    c.notes('pad', 89, [n('G3'), n('Bb3'), n('D4')], 1, 5.5, 50)
    return render('m_title', c, tail=5.0, room=0.92, wet=0.38, target_rms=-18, loop=False)


@cue
def m_movement():
    """Longer sting for movement cards."""
    c = Cue(60, 16)
    c.note('timp', 47, n('D2'), 0, 1, 70)
    for k in range(24):
        c.note('timp', 47, n('D2'), k * 0.25, 0.25, 30 + k * 3)
    c.notes('hit', 55, [n('D2'), n('A2'), n('D3')], 6, 1.5, 110)
    c.notes('brass', 61, chord('D', 'm', 3), 6, 5, 90)
    c.notes('str', 48, [n('D2')] + chord('D', 'm', 3) + [n('D5')], 6, 6, 90)
    c.notes('choir', 52, chord('D', 'm', 4), 6, 6, 80)
    motif(c, 'piano', 0, 7.5, octave=5, root='D', vel=72, dur=0.9)
    c.swell('str', 48, 0, 6, 40, 110)
    return render('m_movement', c, tail=6.0, room=0.93, wet=0.38, target_rms=-18, loop=False)


@cue
def m_corporate():
    c = Cue(104, 17)
    bars = 40
    prog = [('C', 'm'), ('Ab', 'M'), ('Eb', 'M'), ('Bb', 'M')]
    for bar in range(bars):
        b = bar * 4
        root, q = prog[(bar // 2) % 4]
        ch = chord(root, q, 3)
        for k in range(8):
            c.note('pluck', 46 if bar < 8 else 0, sorted(ch)[k % 3] + 12, b + k * 0.5, 0.35, 55)
        if bar % 2 == 0:
            c.notes('pad', 90, ch, b, 8.1, 42)
            c.note('bass', 38, ch[0] - 12, b, 7.5, 60)
        if bar >= 8:
            c.note('kick', 0, 36, b, 0.3, 70, drum=True)
            c.note('kick', 0, 36, b + 2, 0.3, 60, drum=True)
            for k in range(8):
                c.note('hat', 0, 42, b + k * 0.5, 0.1, 35 + (10 if k % 2 else 0), drum=True)
    return render('m_corporate', c, room=0.6, wet=0.2, target_rms=-22)


def build(names=None, force=False):
    os.makedirs(OUT, exist_ok=True)
    for name in names or list(REG):
        path = os.path.join(OUT, f'{name}.wav')
        if os.path.exists(path) and not force:
            continue
        x = REG[name]()
        print(f'{name:14s} {x.shape[1] / SR:6.1f}s  rms {rms_db(x):6.1f} dB', flush=True)



# ------------------------------------------------------- cues added for the rebuild
# The rebuilt script needs four moods the original score did not have: a quiet bed for the
# many reflective two-handers, a bright one for the reveals, a bell for the moments when
# something is placed in the world, and a cold open.

@cue
def m_lo():
    """Low, quiet, reflective. Cello and a sparse piano; nothing happens in it on purpose."""
    c = Cue(52, 21)
    prog = [('G', 'm'), ('Eb', 'M7'), ('Bb', 'M'), ('F', 'sus4')]
    bars = 40
    for bar in range(bars):
        b = bar * 4
        root, q = prog[(bar // 2) % 4]
        ch = chord(root, q, 3)
        c.note('cello', 42, ch[0] - 12, b, 8.2 if bar % 2 == 0 else 4.1, 44)
        c.notes('pad', 89, ch, b, 4.2, 34)
        if bar % 8 == 4:
            motif(c, 'piano', 0, b, octave=4, vel=48, dur=1.5)
        elif c.r.random() < 0.4:
            c.note('piano', 0, ch[c.r.integers(0, len(ch))] + 12, b + c.r.uniform(1, 3), 2.0, 38)
    c.swell('cello', 42, 0, bars * 4, 55, 80)
    return render('m_lo', c, room=0.85, wet=0.30, target_rms=-25)


@cue
def m_hi():
    """Wonder. High strings over a rising pad, for the reveals and the things in the sky."""
    c = Cue(64, 22)
    prog = [('D', 'add9'), ('A', 'M'), ('B', 'm9'), ('G', 'M7')]
    bars = 32
    for bar in range(bars):
        b = bar * 4
        root, q = prog[bar % 4]
        ch = chord(root, q, 3)
        c.notes('pad', 91, ch, b, 4.2, 42 + min(30, bar))
        c.note('sub', 38, ch[0] - 12, b, 3.9, 46)
        top = sorted(ch)[-1] + 12
        c.note('str', 48, top, b, 4.1, 40 + min(35, bar))
        if bar >= 8:
            for k in range(4):
                c.note('bells', 14, sorted(ch)[k % len(ch)] + 24, b + k * 1.0, 0.9, 30 + (k % 2) * 6)
        if bar >= 20:
            c.notes('brass', 61, [ch[0], ch[2]], b, 4.0, 38 + (bar - 20) * 2)
    c.swell('str', 48, 0, bars * 4, 50, 110)
    return render('m_hi', c, room=0.92, wet=0.34, target_rms=-21)


@cue
def m_bell():
    """One slow bell figure with a very long decay: something has just been placed in the world."""
    c = Cue(44, 23)
    bars = 24
    for bar in range(bars):
        b = bar * 4
        ch = chord('G', 'm' if bar % 4 < 2 else 'M7', 2)
        c.notes('drone', 89, [ch[0], ch[0] + 7], b, 8.2, 30)
        if bar % 2 == 0:
            c.note('bell', 14, n('G5') if bar % 8 == 0 else n('D5'), b, 3.8, 58 - (bar // 2))
            c.note('bell', 14, n('Bb4'), b + 1.5, 2.5, 40)
        if bar == 8 or bar == 16:
            motif(c, 'bell', 14, b, octave=5, vel=52, dur=2.0)
    return render('m_bell', c, room=0.95, wet=0.42, target_rms=-24)


@cue
def m_open():
    """The cold open: a drone, the motif at the bottom of the piano, and nothing else."""
    c = Cue(48, 24)
    bars = 20
    for bar in range(bars):
        b = bar * 4
        c.note('drone', 89, n('G1'), b, 4.2, 34)
        c.note('drone', 89, n('D2'), b, 4.2, 26)
        if bar in (4, 12):
            motif(c, 'piano', 0, b, octave=3, vel=54, dur=1.5)
        if bar >= 8 and bar % 4 == 2:
            c.note('str', 49, n('Eb3'), b, 4.0, 30 + bar)
        if bar >= 14:
            c.note('sub', 38, n('G1'), b, 3.9, 40 + (bar - 14) * 3)
    c.swell('str', 49, 32, bars * 4, 30, 80)
    return render('m_open', c, room=0.9, wet=0.36, target_rms=-24)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    build(args or None, force='--force' in sys.argv)
