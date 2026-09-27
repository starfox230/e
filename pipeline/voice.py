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
VERSION = 'v4'   # asterisk stripping, new pronunciation table, Piper support

# speaker -> (kokoro voice, speed, fx preset, level offset dB)
VOICES = {
    'N':         ('am_michael', 0.94, 'narrator', 0.0),
    'WESKER':    ('piper:lessac', 0.64, 'wesker', 0.0),
    'ADOLF':     ('piper:alan', 0.72, 'adolf', -1.5),
    'SYSTEM':    ('af_nicole', 0.98, 'system', -1.0),
    'COUNCILLOR': ('am_liam*0.5+am_adam*0.5', 0.97, 'dialog', 0.0),
    'REISS':     ('am_puck*0.5+am_adam*0.5', 0.95, 'dialog', 0.0),
    'DOCTOR':    ('af_aoede*0.6+af_kore*0.4', 0.96, 'dialog', 0.0),
    'TEACHER':   ('af_sarah*0.6+af_nova*0.4', 0.97, 'dialog', 0.0),
    'CHILD':     ('af_nicole*0.4+bf_lily*0.6', 1.04, 'dialog', 0.0),
    'TESH':      ('bf_isabella*0.5+af_aoede*0.5', 0.88, 'alien_bright', 0.0),
    'MIRA':      ('af_nova*0.5+af_kore*0.5', 0.98, 'dialog', 0.0),
    'COLONEL':   ('bm_lewis*0.4+am_eric*0.6', 0.95, 'dialog', 0.0),
    'BANKER':    ('bm_daniel*0.5+am_eric*0.5', 0.96, 'dialog', 0.0),
    'CHIEF':     ('am_onyx*0.6+am_eric*0.4', 0.94, 'dialog', 0.0),
    'MAYOR':     ('af_sarah*0.5+bf_alice*0.5', 0.95, 'dialog', 0.0),
    'ANALYST':   ('am_eric', 1.0, 'broadcast', 0.0),
    'OFFICER':   ('am_echo', 0.98, 'radio', 0.0),
    'MINISTER':  ('bm_daniel', 0.95, 'dialog', 0.0),
    'GENERAL':   ('am_onyx', 0.92, 'dialog', 0.0),
    'DIRECTOR':  ('bf_emma', 0.96, 'dialog', 0.0),
    'VOSS':      ('bf_alice', 0.95, 'dialog', 0.0),
    'BIRKIN':    ('am_puck', 1.06, 'dialog', 0.0),
    'ANNETTE':   ('af_kore*0.5+bf_alice*0.5', 0.97, 'dialog', 0.0),
    'HUNK':      ('am_echo', 0.94, 'radio', 0.0),
    'REDQUEEN':  ('af_nicole*0.5+bf_lily*0.5', 1.0, 'system', 0.0),
    'MARGUERITE': ('af_sarah', 1.0, 'flat', 0.0),
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
    'VETH':      ('bf_isabella*0.5+bm_george*0.5', 0.85, 'alien_deep', 0.0),
    'SETHRA':    ('bf_isabella*0.6+bm_george*0.4', 0.86, 'alien_deep', 0.0),
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
    'SENATOR':   ('af_kore*0.5+bf_alice*0.5', 0.96, 'dialog', 0.0),
    'DENISE':    ('af_nova*0.5+af_sarah*0.5', 0.97, 'dialog', 0.0),
    'ELEANOR':   ('af_aoede*0.5+af_sarah*0.5', 0.94, 'dialog', 0.0),
    'ORACLE':    ('af_sky*0.5+am_echo*0.5', 1.0, 'broadcast', 0.0),
    'MIREILLE':  ('af_nicole*0.5+ff_siwis*0.5', 0.98, 'dialog', 0.0),
    'DANILO':    ('am_liam*0.4+pm_alex*0.6', 0.97, 'dialog', 0.0),
}

HINT_SPEED = {'phone': 1.0, 'slow': 0.9, 'fast': 1.08, 'soft': 0.95, 'whisper': 0.93, 'shout': 1.04, 'cold': 0.95}
HINT_GAIN = {'soft': -3.0, 'whisper': -5.0, 'shout': 2.0}

PRON = {
    # Respellings for the names and terms the synthesisers get wrong. Both engines are good at
    # ordinary English; what they guess at is Igbo, Yoruba, Mandarin, Kiribati and invented
    # words, and a mispronounced character name is noticed every single time it recurs.
    r'\bAdeyemi\b': 'Ah-deh-YEH-mee', r'\bOkonkwo\b': 'Oh-KON-kwoh', r'\bOdhiambo\b': 'Oh-dee-AM-boh',
    r'\bAchieng\b': 'Ah-chee-ENG', r'\bAdaeze\b': 'Ah-dah-EH-zeh', r'\bIyabo\b': 'Ee-YAH-boh',
    r'\bKalu\b': 'KAH-loo', r'\bAbayomi\b': 'Ah-bah-YOH-mee',
    r'\bWen Shulan\b': 'Wen Shoo-LAHN', r'\bShulan\b': 'Shoo-LAHN',
    r'\bHangzhou\b': 'Hahng-joh', r'\bSuzhou\b': 'Soo-joh', r'\bBeihai\b': 'Bay-hi',
    r'\bKisumu\b': 'Kih-SOO-moo', r'\bTeraina\b': 'Teh-rah-EE-nah', r'\bRabaere\b': 'Rah-bah-EH-reh',
    r'\bOchoa\b': 'Oh-CHOH-ah', r'\bBastida\b': 'Bah-STEE-dah', r'\bFerrand\b': 'Feh-RAHN',
    r'\bKowal\b': 'KOH-val', r'\bChandrasekhar\b': 'Chan-druh-SAY-kar', r'\bRinaldi\b': 'Ree-NAHL-dee',
    r'\bBonetti\b': 'Boh-NET-tee', r'\bMarguerite\b': 'Mar-guh-REET', r'\bAnnette\b': 'Ah-NET',
    r'\bCatherine\b': 'KATH-rin', r'\bEvelyn\b': 'EV-uh-lin', r'\bWesker\b': 'WES-ker',
    r'\bBirkin\b': 'BUR-kin', r'\bCarrick\b': 'KA-rick', r'\bTolliver\b': 'TOL-ih-ver',
    r'\bHalder\b': 'HAHL-der', r'\bTesh\b': 'Tesh', r'\bVoss\b': 'Voss', r'\bReiss\b': 'Rice',
    r'\bHUNK\b': 'Hunk', r'\bPell\b': 'Pell', r'\bSloane\b': 'Slone', r'\bWyatt\b': 'WY-ut',
    r'\bArklay\b': 'ARK-lay', r'\bAeterna\b': 'Ay-TUR-nah', r'\bProgenitor\b': 'proh-JEN-ih-tor',
    r'\bVeth\b': 'Veth', r'\bSethra-Ka\b': 'SETH-rah Kah', r'\bAshet\b': 'AH-shet',
    r'\bMyriad\b': 'MIH-ree-ad', r'\bTollhouse\b': 'Toll-house', r'\bKelpie\b': 'KEL-pee',
    r'\bRavenna\b': 'Rah-VEN-nah', r'\bBasel\b': 'BAH-zul', r'\bLyon\b': 'Lee-ON',
    r'\bWilmington\b': 'WIL-ming-ton', r'\bChristina\b': 'Kris-TEE-nah',
    r'\bAllegheny\b': 'Al-uh-GAY-nee', r'\bDelaware\b': 'DEL-uh-ware',
    r'\bLANTERN\b': 'Lantern', r'\bORIGIN\b': 'Origin', r'\bBEYOND\b': 'Beyond',
    # abbreviations and units
    r'\bB\.O\.W\.s\b': 'B O Ws', r'\bB\.O\.W\.\b': 'B O W', r'\bBSL-(\d)': r'B S L \1',
    r'\bT-103\b': 'T one oh three', r'\bT-00\b': 'T zero zero',
    r'\bDr\.': 'Doctor', r'\bMr\.': 'Mister', r'\bMrs\.': 'Missus', r'\bMs\.': 'Miz',
    r'\bSt\. ': 'Saint ', r'\ba\.m\.': 'A M', r'\bp\.m\.': 'P M', r'\bvs\.': 'versus',
    r'\bU\.S\.': 'U S', r'\bU\.K\.': 'U K', r'\bOk\b': 'okay',
    r'\bWHO\b': 'W H O', r'\bGAO\b': 'G A O', r'\bFBI\b': 'F B I', r'\bCIA\b': 'C I A',
    r'\bMSS\b': 'M S S', r'\bNATO\b': 'NAY-toh', r'\bU\.S\.S\.\b': 'U S S',
    r'\bL4\b': 'L four', r'\bB4\b': 'B four',
}

_kokoro = None
_piper = {}

PIPER_DIR = '/home/user/e/work/models/piper'
PIPER_VOICES = {
    'ryan':   'en/en_US/ryan/high/en_US-ryan-high.onnx',
    'lessac': 'en/en_US/lessac/high/en_US-lessac-high.onnx',
    'alan':   'en/en_GB/alan/medium/en_GB-alan-medium.onnx',
}


def piper(name):
    """A Piper voice, loaded once. Piper is slower than Kokoro (about two times realtime on
    four cores) and clearer: it phonemises through espeak-ng, so it does not mispronounce the
    names and the abbreviations that Kokoro guesses at."""
    if name not in _piper:
        from piper import PiperVoice
        _piper[name] = PiperVoice.load(os.path.join(PIPER_DIR, PIPER_VOICES[name]))
    return _piper[name]


def piper_say(name, text, speed=1.0, sentence_pause=0.30, clause_pause=0.10):
    """Mono float32 at this module's sample rate.

    Piper renders a whole paragraph at one pace with no breath in it, which is why its lines
    came out a third faster than Kokoro's narration even at the same nominal speed. Sentences
    are synthesised separately and rejoined with a pause, the way Kokoro does it internally,
    so a character's delivery breathes instead of running on.
    """
    import io
    import wave
    v = piper(name)
    try:
        from piper import SynthesisConfig
        cfg = SynthesisConfig(length_scale=1.0 / speed)
    except Exception:
        cfg = None
    parts = [p for p in re.split(r'(?<=[.!?])\s+', text.strip()) if p]
    out = []
    for k, part in enumerate(parts):
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as w:
            v.synthesize_wav(part, w, syn_config=cfg) if cfg else v.synthesize_wav(part, w)
        buf.seek(0)
        a, sr = sf.read(buf, dtype='float32')
        if a.ndim > 1:
            a = a.mean(axis=1)
        a = (signal.resample_poly(a, SR, sr) if sr != SR else a).astype(np.float32)
        out.append(a)
        if k < len(parts) - 1:
            gap = sentence_pause if part.endswith(('.', '!', '?')) else clause_pause
            out.append(np.zeros(int(gap * SR), dtype=np.float32))
    return (np.concatenate(out) if out else np.zeros(1, dtype=np.float32)), SR


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
    # the scripts mark quoted documents and emphasis with asterisks; neither engine knows
    # what to do with one, and Kokoro reads some of them aloud
    s = re.sub(r'\*([^*]+)\*', r'\1', s)
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
        # a closed door: down two and a half semitones, weight under it, and the top rolled
        # off so nothing in the voice is ever bright
        return Pedalboard([PitchShift(semitones=-2.5), HighpassFilter(cutoff_frequency_hz=80),
                           LowShelfFilter(cutoff_frequency_hz=150, gain_db=3.0),
                           Compressor(threshold_db=-22, ratio=3.0, attack_ms=4, release_ms=110),
                           HighShelfFilter(cutoff_frequency_hz=6500, gain_db=-2.5),
                           Reverb(room_size=0.24, wet_level=0.06, dry_level=1.0, damping=0.75)])
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
        # a clip written while the disk was full has an intact header and no data behind it,
        # and reading it either fails or asks numpy for a terabyte; treat it as a cache miss
        try:
            info = sf.info(path)
            if 0 < info.frames / info.samplerate <= 120 and os.path.getsize(path) > info.frames * 0.02:
                return path
        except Exception:
            pass
        os.remove(path)
    if voice.startswith('piper:'):
        audio, sr = piper_say(voice.split(':', 1)[1], spoken, speed)
    else:
        lang = 'en-gb' if voice.startswith('b') else 'en-us'
        audio, sr = kokoro().create(spoken, voice=style(voice), speed=speed, lang=lang, sentence_pause=0.32, clause_pause=0.12)
        audio = (signal.resample_poly(audio.astype(np.float32), SR // sr, 1) if SR % sr == 0
                 else signal.resample(audio, int(len(audio) * SR / sr)))
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
