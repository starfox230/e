"""Multi-voice narration with Kokoro-82M (ONNX) plus per-character processing. Clips are cached by content hash."""
import os
import re
import hashlib
import numpy as np
import soundfile as sf
from scipy import signal
from num2words import num2words
from pedalboard import (Pedalboard, PitchShift, HighpassFilter, LowpassFilter, Compressor, Reverb, Chorus,
                        LowShelfFilter, HighShelfFilter, Gain, Delay)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, 'work', 'tts')
SR = 48000
VERSION = 'v3'

# speaker -> (kokoro voice, speed, fx preset, level offset dB)
VOICES = {
    'N':         ('am_michael', 0.94, 'narrator', 0.0),
    'WESKER':    ('am_fenrir', 0.88, 'wesker', 0.0),
    'ADOLF':     ('bm_george', 0.86, 'adolf', -1.5),
    'SYSTEM':    ('af_nicole', 0.98, 'system', -1.0),
    'WALTER':    ('am_onyx', 0.90, 'dialog', 0.0),
    'DANA':      ('af_sarah', 0.97, 'dialog', 0.0),
    'LESSING':   ('bm_lewis', 0.93, 'dialog', 0.0),
    'EVELYN':    ('af_bella', 0.98, 'dialog', 0.0),
    'TOMAS':     ('am_puck', 1.02, 'dialog', 0.0),
    'NADIA':     ('af_aoede', 0.95, 'dialog', 0.0),
    'CATHERINE': ('bf_emma', 0.96, 'dialog', 0.0),
    'PELL':      ('bm_daniel', 0.90, 'flat', 0.0),
    'HALE':      ('am_echo', 0.88, 'flat', 0.0),
    'GUS':       ('am_santa', 0.95, 'flat', 0.0),
    'HANNAH':    ('af_nova', 1.00, 'dialog', 0.0),
    'TOLLIVER':  ('am_eric', 0.98, 'dialog', 0.0),
    'KUCHAR':    ('am_liam', 1.02, 'dialog', 0.0),
    'DELGADO':   ('am_adam', 0.97, 'dialog', 0.0),
    'CARRICK':   ('am_michael', 1.00, 'carrick', 0.0),
    'HADDAD':    ('af_kore', 0.97, 'dialog', 0.0),
    'SLOANE':    ('am_adam', 0.95, 'dialog', 0.0),
    'ASHCOMBE':  ('bm_fable', 0.95, 'dialog', 0.0),
    'HALVORSEN': ('bf_isabella', 0.95, 'dialog', 0.0),
    'WYATT':     ('am_onyx', 0.92, 'dialog', 0.0),
    'CRANE':     ('am_liam', 1.00, 'dialog', 0.0),
    'WEN':       ('af_sky*0.6+bf_alice*0.4', 0.92, 'dialog', 0.0),
    'BELOV':     ('bm_lewis', 0.90, 'dialog', 0.0),
    'ANCHOR':    ('af_jessica', 1.00, 'broadcast', 0.0),
    'REPORTER':  ('am_eric', 1.02, 'broadcast', 0.0),
    'MAN':       ('am_liam', 1.00, 'dialog', 0.0),
    'WOMAN':     ('af_river', 1.00, 'dialog', 0.0),
    'SOLDIER':   ('am_echo', 1.02, 'radio', 0.0),
    'RADIO':     ('am_adam', 1.02, 'radio', 0.0),
    'CROWD':     ('am_liam', 1.00, 'dialog', 0.0),
    'VETH':      ('bm_george', 0.85, 'alien_deep', 0.0),
    'BROKER':    ('af_alloy', 0.98, 'alien_bright', 0.0),
    'CHOIR':     ('af_river', 0.90, 'alien_choir', 0.0),
    'MYRIAD':    ('am_echo', 0.90, 'alien_hive', 0.0),
    'PRESIDENT': ('af_heart*0.5+bf_emma*0.5', 0.93, 'dialog', 0.0),
    'FENWICK':   ('bm_fable', 0.99, 'dialog', 0.0),
    'IMOGEN':    ('bf_lily', 0.95, 'dialog', 0.0),
    'TAN':       ('af_river*0.5+af_sky*0.5', 1.0, 'dialog', 0.0),
    'ELLERY':    ('bm_daniel*0.5+bm_george*0.5', 0.84, 'dialog', 0.0),
    'DORSEY':    ('am_eric*0.5+am_onyx*0.5', 0.88, 'dialog', 0.0),
    'KOWAL':     ('af_heart', 0.97, 'dialog', 0.0),
    'FERRIS':    ('am_liam*0.5+am_eric*0.5', 1.0, 'dialog', 0.0),
    'LAURA':     ('af_sarah*0.5+af_heart*0.5', 0.98, 'dialog', 0.0),
    'MAUER':     ('bf_isabella*0.5+af_kore*0.5', 0.95, 'dialog', 0.0),
    'FERRAND':   ('bm_lewis*0.4+am_eric*0.6', 0.95, 'dialog', 0.0),
    'LAWYER':    ('am_eric*0.4+am_adam*0.6', 1.0, 'dialog', 0.0),
    'ELEANOR':   ('af_aoede*0.5+af_sarah*0.5', 0.94, 'dialog', 0.0),
    'ORACLE':    ('af_sky*0.5+am_echo*0.5', 1.0, 'broadcast', 0.0),
    'MIREILLE':  ('af_nicole*0.5+ff_siwis*0.5', 0.98, 'dialog', 0.0),
    'DANILO':    ('am_liam*0.4+pm_alex*0.6', 0.97, 'dialog', 0.0),
}

HINT_SPEED = {'phone': 1.0, 'slow': 0.9, 'fast': 1.08, 'soft': 0.95, 'whisper': 0.93, 'shout': 1.04, 'cold': 0.95}
HINT_GAIN = {'soft': -3.0, 'whisper': -5.0, 'shout': 2.0}

PRON = {
    r'\bTomás\b': 'Tomas', r'\bAdeyemi\b': 'Addeh-yemmy', r'\bKuchar\b': 'Koo-car', r'\bAeterna\b': 'Ay-turna',
    r'\bTianhe\b': 'Tyen-huh', r'\bShulan\b': 'Shoo-lahn', r'\bBelov\b': 'Byeh-loff', r'\bOkafor\b': 'Oh-kah-for',
    r'\bB\.O\.W\.s\b': 'B O Ws', r'\bB\.O\.W\.\b': 'B O W', r'\bBSL-(\d)': r'B S L \1', r'\bT-00\b': 'T zero zero',
    r'\bDr\.': 'Doctor', r'\bMr\.': 'Mister', r'\bMrs\.': 'Missus', r'\bMs\.': 'Miz', r'\bSt\. ': 'Saint ',
    r'\ba\.m\.': 'A M', r'\bp\.m\.': 'P M', r'\bvs\.': 'versus', r'\bU\.S\.': 'U S', r'\bU\.K\.': 'U K',
    r'\bHaddad\b': 'Ha-dahd', r'\bAshcombe\b': 'Ash-cum', r'\bHalvorsen\b': 'Hal-vor-sen', r'\bVeth\b': 'Veth',
    r'\bWen\b': 'Wen', r'\bLessing\b': 'Lessing', r'\bArklay\b': 'Ark-lay', r'\bOk\b': 'okay',
}

_kokoro = None


def kokoro():
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro
        _kokoro = Kokoro('/opt/models/kokoro/kokoro-v1.0.onnx', '/opt/models/kokoro/voices-v1.0.bin')
    return _kokoro


def _n(txt):
    v = float(txt.replace(',', ''))
    return num2words(int(v)) if v == int(v) else num2words(v)


def style(spec):
    """'am_michael' or a blend such as 'bm_daniel*0.5+bm_george*0.5'."""
    if '+' not in spec and '*' not in spec:
        return spec
    k = kokoro()
    total = None
    for part in spec.split('+'):
        name, w = (part.split('*') + ['1'])[:2]
        v = k.get_voice_style(name.strip()) * float(w)
        total = v if total is None else total + v
    return total


def normalize_text(text):
    s = text
    for pat, rep in PRON.items():
        s = re.sub(pat, rep, s)
    # currency
    def money(m):
        num = float(m.group(1).replace(',', ''))
        unit = (m.group(2) or '').strip().lower()
        mult = {'k': ' thousand', 'm': ' million', 'b': ' billion', 't': ' trillion',
                'million': ' million', 'billion': ' billion', 'trillion': ' trillion', 'thousand': ' thousand'}.get(unit, '')
        words = num2words(num).replace(' point zero', '') if num != int(num) else num2words(int(num))
        return f'{words}{mult} dollars'
    s = re.sub(r'\$(\d[\d,]*(?:\.\d+)?)(?:\s*(k|m|b|t|million|billion|trillion|thousand)\b)?', money, s, flags=re.I)
    s = re.sub(r'£(\d[\d,]*(?:\.\d+)?)(?:\s*(million|billion|thousand)\b)?', lambda m: f"{_n(m.group(1))}{' ' + m.group(2) if m.group(2) else ''} pounds", s)
    s = re.sub(r'\b(\d{1,2}):(\d{2})\b', lambda m: f"{num2words(int(m.group(1)))} {'o ' + num2words(int(m.group(2))) if m.group(2).startswith('0') and m.group(2) != '00' else ('' if m.group(2) == '00' else num2words(int(m.group(2))))}".strip(), s)
    s = re.sub(r'(\d+(?:\.\d+)?)%', lambda m: f'{num2words(float(m.group(1)) if "." in m.group(1) else int(m.group(1)))} percent', s)
    s = re.sub(r'\b(1[89]\d\d|20\d\d)\b', lambda m: num2words(int(m.group(1)), to='year'), s)
    s = re.sub(r'\b\d{1,3}(?:,\d{3})+\b', lambda m: num2words(int(m.group(0).replace(',', ''))), s)
    s = re.sub(r'\b\d+\.\d+\b', lambda m: num2words(float(m.group(0))), s)
    s = re.sub(r'\b\d+\b', lambda m: num2words(int(m.group(0))), s)
    s = s.replace('&', ' and ')
    return s


def _board(preset):
    base = [HighpassFilter(cutoff_frequency_hz=70), Compressor(threshold_db=-20, ratio=2.5, attack_ms=5, release_ms=90)]
    if preset == 'narrator':
        return Pedalboard([LowShelfFilter(cutoff_frequency_hz=180, gain_db=2.0), *base,
                           HighShelfFilter(cutoff_frequency_hz=7000, gain_db=1.0),
                           Reverb(room_size=0.18, wet_level=0.05, dry_level=1.0, damping=0.7)])
    if preset == 'wesker':
        return Pedalboard([PitchShift(semitones=-1.0), LowShelfFilter(cutoff_frequency_hz=160, gain_db=2.5), *base,
                           Reverb(room_size=0.22, wet_level=0.07, dry_level=1.0, damping=0.6)])
    if preset == 'adolf':
        return Pedalboard([PitchShift(semitones=-2.5), LowpassFilter(cutoff_frequency_hz=5200), *base,
                           Delay(delay_seconds=0.028, feedback=0.1, mix=0.18),
                           Reverb(room_size=0.78, wet_level=0.26, dry_level=0.85, damping=0.55, width=1.0)])
    if preset == 'system':
        return Pedalboard([HighpassFilter(cutoff_frequency_hz=280), LowpassFilter(cutoff_frequency_hz=7500), *base,
                           Chorus(rate_hz=0.7, depth=0.18, mix=0.25),
                           Reverb(room_size=0.35, wet_level=0.14, dry_level=0.95, damping=0.3)])
    if preset == 'carrick':
        return Pedalboard([PitchShift(semitones=0.8), *base, Reverb(room_size=0.2, wet_level=0.06, dry_level=1.0)])
    if preset == 'flat':
        return Pedalboard([*base, LowpassFilter(cutoff_frequency_hz=9000),
                           Reverb(room_size=0.2, wet_level=0.05, dry_level=1.0)])
    if preset == 'broadcast':
        return Pedalboard([HighpassFilter(cutoff_frequency_hz=180), LowpassFilter(cutoff_frequency_hz=8000),
                           Compressor(threshold_db=-24, ratio=4, attack_ms=3, release_ms=60)])
    if preset == 'radio':
        return Pedalboard([HighpassFilter(cutoff_frequency_hz=400), LowpassFilter(cutoff_frequency_hz=3200),
                           Compressor(threshold_db=-26, ratio=6, attack_ms=2, release_ms=50), Gain(gain_db=3)])
    if preset == 'alien_deep':
        return Pedalboard([PitchShift(semitones=-6), *base, Chorus(rate_hz=0.3, depth=0.5, mix=0.4),
                           Reverb(room_size=0.9, wet_level=0.3, dry_level=0.8)])
    if preset == 'alien_bright':
        return Pedalboard([PitchShift(semitones=3), *base, Chorus(rate_hz=2.5, depth=0.3, mix=0.35),
                           Reverb(room_size=0.5, wet_level=0.15, dry_level=0.9)])
    if preset == 'alien_choir':
        return Pedalboard([*base, Chorus(rate_hz=0.5, depth=0.8, mix=0.6), Delay(delay_seconds=0.12, feedback=0.3, mix=0.25),
                           Reverb(room_size=0.97, wet_level=0.45, dry_level=0.6)])
    if preset == 'alien_hive':
        return Pedalboard([PitchShift(semitones=-3), *base, Chorus(rate_hz=6.0, depth=0.6, mix=0.5),
                           Reverb(room_size=0.7, wet_level=0.25, dry_level=0.8)])
    return Pedalboard([*base, Reverb(room_size=0.2, wet_level=0.06, dry_level=1.0, damping=0.6)])


def _trim(x, thr_db=-48, pad=0.03):
    thr = 10 ** (thr_db / 20)
    a = np.abs(x).max(axis=0)
    idx = np.where(a > thr)[0]
    if len(idx) == 0:
        return x
    s = max(0, idx[0] - int(pad * SR))
    e = min(x.shape[1], idx[-1] + int(pad * SR * 3))
    return x[:, s:e]


def _active_rms_db(x):
    m = x.mean(axis=0)
    w = int(0.05 * SR)
    k = len(m) // w
    if k == 0:
        return -60.0
    frames = m[:k * w].reshape(k, w)
    r = np.sqrt((frames ** 2).mean(axis=1))
    act = r[r > 10 ** (-40 / 20)]
    if len(act) == 0:
        return -60.0
    return 20 * np.log10(np.sqrt((act ** 2).mean()) + 1e-12)


def synth(speaker, text, hint=''):
    """Return path to a processed 48 kHz stereo WAV clip for this line (cached)."""
    voice, speed, preset, level = VOICES.get(speaker, VOICES['N'])
    speed *= HINT_SPEED.get(hint, 1.0)
    level += HINT_GAIN.get(hint, 0.0)
    spoken = normalize_text(text)
    key = hashlib.sha1((f'{VERSION}|{voice}|{speed:.3f}|{preset}|{level}|{spoken}' + (f'|{hint}' if hint == 'phone' else '')).encode()).hexdigest()[:16]
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f'{key}.flac')
    if os.path.exists(path):
        return path
    lang = 'en-gb' if voice.startswith('b') else 'en-us'
    audio, sr = kokoro().create(spoken, voice=style(voice), speed=speed, lang=lang, sentence_pause=0.32, clause_pause=0.12)
    audio = signal.resample_poly(audio.astype(np.float32), SR // sr, 1) if SR % sr == 0 else signal.resample(audio, int(len(audio) * SR / sr))
    x = np.stack([audio, audio]).astype(np.float32)
    if hint == 'phone':
        x = Pedalboard([HighpassFilter(cutoff_frequency_hz=320), LowpassFilter(cutoff_frequency_hz=3400),
                        Compressor(threshold_db=-24, ratio=5, attack_ms=2, release_ms=60), Gain(gain_db=2)])(x, SR)
    else:
        x = _board(preset)(x, SR)
    x = _trim(x)
    target = -20.0 + level
    x *= 10 ** ((target - _active_rms_db(x)) / 20)
    peak = np.abs(x).max()
    if peak > 0.95:
        x *= 0.95 / peak
    sf.write(path, x.T, SR, subtype='PCM_16')
    return path


if __name__ == '__main__':
    import sys
    print(normalize_text(' '.join(sys.argv[1:]) or 'Dr. Wesker paid $1.2 million in 2026 — 71% of it, 1,500 Points.'))
