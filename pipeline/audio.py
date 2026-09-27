"""Build a chapter's timeline from its script and mix the soundtrack.

Every time value comes from measured clip lengths, so the timeline JSON that drives the video
is exactly the same clock as the audio it is muxed with.

Usage: python3 pipeline/audio.py script/ch01.txt
Outputs: work/chNN/mix.wav, work/chNN/timeline.json, out/subs/chNN.srt
"""
import os
import sys
import json
import hashlib
import numpy as np
import soundfile as sf
import pyloudnorm
from scipy import signal
from parse import parse
from voice import synth, SR, VOICES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SFX = os.path.join(ROOT, 'assets', 'sfx')
MUSIC = os.path.join(ROOT, 'assets', 'music')

GAP_NARR = 0.38        # after a narration paragraph
GAP_DIALOG = 0.26      # after a line of dialogue
GAP_SWITCH = 0.12      # extra when the speaker changes
MUSIC_GAP_DB = -26.0   # music level (RMS dBFS) when nobody is speaking
MUSIC_DUCK_DB = -8.0   # extra reduction under speech
AMB_DB = -35.0
AMB_DUCK_DB = -3.0
SFX_DB = -23.0         # one-shot active RMS
# Per-effect trims on top of SFX_DB. Every one-shot is otherwise levelled to the same loudness,
# which is right for a door or a gunshot and wrong for an interface tone: the System's panel
# chime sat at gunshot level. It is a small ding under the voice now, not an event.
SFX_TRIM = {'sfx_system_open': -11.0, 'sfx_system_chime': -11.0}
TARGET_LUFS = -15.0
CEILING_DB = -1.5
STINGS = {'m_title', 'm_movement'}

_cache = {}


def load(path):
    if path not in _cache:
        y, s = sf.read(path, always_2d=True, dtype='float32')
        y = y.T
        if y.shape[0] == 1:
            y = np.concatenate([y, y])
        if s != SR:
            y = signal.resample_poly(y, SR, s, axis=1).astype(np.float32)
        _cache[path] = y
    return _cache[path]


def active_rms_db(x, floor=-45):
    m = np.abs(x).mean(axis=0)
    w = int(0.05 * SR)
    k = len(m) // w
    if k == 0:
        return -60.0
    r = np.sqrt((x[:, :k * w].reshape(2, k, w) ** 2).mean(axis=(0, 2)))
    act = r[r > 10 ** (floor / 20)]
    return 20 * np.log10(np.sqrt((act ** 2).mean()) + 1e-12) if len(act) else -60.0


def looped(x, n, xfade=4.0):
    """Return n samples of x looped with crossfades."""
    xf = int(xfade * SR)
    if x.shape[1] >= n:
        return x[:, :n]
    out = np.zeros((2, n), np.float32)
    pos = 0
    ramp = np.linspace(0, 1, xf, dtype=np.float32)
    first = True
    while pos < n:
        seg = x.copy()
        if not first:
            seg[:, :xf] *= ramp
        seg[:, -xf:] *= ramp[::-1]
        m = min(seg.shape[1], n - pos)
        out[:, pos:pos + m] += seg[:, :m]
        pos += seg.shape[1] - xf
        first = False
    return out


def peak_limit(x, ceiling_db=-1.5, look_ms=12.0):
    """Look-ahead brickwall: min-filter the required gain over 2L, then average over L (never overshoots)."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    c = 10 ** (ceiling_db / 20)
    peak = np.abs(x).max(axis=0)
    need = np.minimum(1.0, c / np.maximum(peak, 1e-9)).astype(np.float32)
    L = int(look_ms / 1000 * SR)
    g = minimum_filter1d(need, size=2 * L + 1, mode='nearest')
    g = uniform_filter1d(g, size=L, mode='nearest')
    y = x * g
    return np.clip(y, -c, c).astype(np.float32)


def env_segment(n, fin, fout):
    e = np.ones(n, np.float32)
    a, b = min(int(fin * SR), n), min(int(fout * SR), n)
    if a:
        e[:a] = np.linspace(0, 1, a)
    if b:
        e[n - b:] *= np.linspace(1, 0, b)
    return e


def shot_id(chapter, prompt):
    return f'c{chapter:02d}_' + hashlib.sha1(prompt.encode()).hexdigest()[:10]


# Pacing. The first cut paused far too often: 31 minutes of scripted [pause] markers, most of
# them 1.5 s on top of the gap after every line, and another ~37 minutes of dead air inside the
# voice clips themselves, where the TTS stalls at dashes and sentence breaks for up to 1.2 s.
# Scripted pauses keep their relative weight but run at a little over half length, and every
# clip has its internal gaps capped at a natural sentence break.
PAUSE_SCALE = 0.55
PAUSE_MIN = 0.2
CLIP_MAX_GAP = 0.36       # longest silence left inside a spoken line (s)
CLIP_EDGE = 0.05          # silence kept at the head and tail of a line (s)
TIGHT = os.path.join(ROOT, 'work', 'tts_tight')


def pause_len(d):
    return max(PAUSE_MIN, round(d * PAUSE_SCALE, 3))


def tighten(path):
    """The clip with its long internal silences cut down to CLIP_MAX_GAP, cached on disk.

    Silence is judged against the clip's own level (40 dB under its loudest 20 ms), so a quiet
    whispered line is treated the same as a shouted one. Each cut is a 15 ms crossfade."""
    key = hashlib.sha1(f'{path}|{os.path.getmtime(path)}|{CLIP_MAX_GAP}|{CLIP_EDGE}|v1'.encode()).hexdigest()[:16]
    out = os.path.join(TIGHT, key + '.wav')
    if os.path.exists(out):
        return out
    os.makedirs(TIGHT, exist_ok=True)
    x, sr = sf.read(path, dtype='float32')
    mono = x.mean(axis=1) if x.ndim > 1 else x
    win = int(0.02 * sr)
    f = len(mono) // win
    if f < 3:
        return path
    e = 20 * np.log10(np.sqrt((mono[:f * win].reshape(f, win) ** 2).mean(axis=1)) + 1e-9)
    loud = e > e.max() - 40
    idx = np.where(loud)[0]
    if len(idx) == 0:
        return path
    edge = int(CLIP_EDGE * sr)
    first, last = idx[0] * win, (idx[-1] + 1) * win
    keep = []                                   # (start, end) sample ranges to keep
    cur = max(0, first - edge)
    run_start = None
    for k in range(idx[0], idx[-1] + 1):
        if not loud[k]:
            run_start = k if run_start is None else run_start
            continue
        if run_start is not None:
            gap = (k - run_start) * win
            if gap > CLIP_MAX_GAP * sr:
                half = int(CLIP_MAX_GAP * sr / 2)
                keep.append((cur, run_start * win + half))
                cur = k * win - half
            run_start = None
    keep.append((cur, min(len(x), last + edge)))
    fade = int(0.015 * sr)
    pieces = []
    for a, b in keep:
        seg = x[a:b].copy()
        if len(seg) > 2 * fade:
            ramp = np.linspace(0, 1, fade, dtype=np.float32)
            if seg.ndim > 1:
                ramp = ramp[:, None]
            seg[:fade] *= ramp
            seg[-fade:] *= ramp[::-1]
        pieces.append(seg)
    y = np.concatenate(pieces)
    sf.write(out, y, sr)
    return out


def build_timeline(ch):
    missing = sorted({e.speaker for e in ch.events if e.kind == 'line' and e.speaker not in VOICES})
    if missing:
        raise ValueError(f'chapter {ch.number}: no voice assigned for {missing}')
    tl = {'chapter': ch.number, 'title': ch.title, 'movement': ch.movement,
          'voice': [], 'shots': [], 'system': [], 'loc': [], 'cards': [], 'music': [], 'amb': [], 'sfx': []}
    t = 0.6
    pending = []
    last_spk = None
    has_cards_marker = any(e.kind == 'cards' for e in ch.events)

    def add_voice(spk, text, hint, at, subtitle=True):
        path = tighten(synth(spk, text, hint))
        dur = sf.info(path).duration
        tl['voice'].append({'start': round(at, 3), 'dur': round(dur, 3), 'path': os.path.relpath(path, ROOT),
                            'speaker': spk, 'text': text, 'sub': subtitle})
        return dur

    def cards(at):
        # One card and one announcement per chapter. A chapter that opens a Part carries the
        # Part's name as a label on its own card, with the heavier Part sting under it. Giving
        # the Part a card of its own put two title cards back to back, both in title type and
        # both read aloud, so the chapter came across as having two names.
        card = {'type': 'chapter', 'start': at, 'end': at + 6.8, 'num': ch.number, 'title': ch.title}
        if ch.movement:
            card['movement'] = list(ch.movement)
            card['end'] = at + 8.0
            tl['music'].append({'time': at, 'cue': 'm_movement', 'gain': 0.0})
        else:
            tl['music'].append({'time': at, 'cue': 'm_title', 'gain': -2.0})
        tl['cards'].append(card)
        from num2words import num2words
        d = add_voice('N', f'Chapter {num2words(ch.number)}. {ch.title}.', '', at + 1.7, subtitle=False)
        return at + max(card['end'] - at + 0.4, 1.7 + d + 1.6)

    def flush(at, skip_chime=False):
        chimed = False
        for e in pending:
            when = at + e.offset
            if e.kind == 'img':
                tl['shots'].append({'start': round(at, 3), 'prompt': e.text, 'id': shot_id(ch.number, e.text)})
            elif e.kind == 'loc':
                # captions type on silently: a bell under every one of them played a ding
                # in nearly every chapter, which read as a mistake rather than a signature
                tl['loc'].append({'start': round(at + 0.4, 3), 'end': round(at + 6.0, 3), 'text': e.text})
            elif e.kind == 'music':
                tl['music'].append({'time': when, 'cue': e.name, 'gain': e.gain})
            elif e.kind in ('amb', 'amb+'):
                tl['amb'].append({'time': when, 'name': e.name, 'layer': e.kind == 'amb+', 'gain': e.gain})
            elif e.kind == 'sfx':
                tl['sfx'].append({'time': when, 'name': e.name, 'gain': e.gain})
                if e.name in ('sfx_system_open', 'sfx_system_chime', 'sfx_level_up', 'sfx_gacha_gold', 'sfx_gacha_rare',
                              'sfx_gacha_common', 'sfx_points'):
                    chimed = True
        pending.clear()
        return chimed

    if not has_cards_marker:
        t = cards(t)
    for e in ch.events:
        if e.kind in ('img', 'loc', 'music', 'amb', 'amb+', 'sfx'):
            pending.append(e)
        elif e.kind == 'cards':
            flush(t)
            t = cards(t)
        elif e.kind == 'title':
            flush(t)
            tl['cards'].append({'type': 'title', 'start': t, 'end': t + 7.5})
            tl['sfx'].append({'time': t, 'name': 'sfx_title_hit', 'gain': 2.0})
            tl['music'].append({'time': t + 0.2, 'cue': 'm_title', 'gain': 0.0})
            t += 8.3
        elif e.kind == 'pause':
            flush(t)
            t += pause_len(e.dur)
        elif e.kind == 'line':
            flush(t)
            if last_spk is not None and last_spk != e.speaker:
                t += GAP_SWITCH
            d = add_voice(e.speaker, e.text, e.hint, t)
            t += d + (GAP_NARR if e.speaker == 'N' else GAP_DIALOG)
            last_spk = e.speaker
        elif e.kind == 'system':
            chimed = flush(t)
            if not chimed:
                tl['sfx'].append({'time': t, 'name': 'sfx_system_chime', 'gain': 0.0})
            say = e.say or e.name.title()
            vstart = t + 0.75
            d = add_voice('SYSTEM', say, '', vstart)
            chars = sum(len(x) for x in e.lines) + len(e.name)
            end = max(vstart + d + 1.1, t + 2.2 + chars * 0.045)
            tl['system'].append({'start': round(t, 3), 'end': round(end, 3), 'title': e.name, 'lines': e.lines})
            # The panel stays up long enough to read, but the soundtrack no longer waits for it:
            # narration resumes as soon as the System has spoken. Holding the audio for the
            # reading time put up to 18 s of silence under every long panel.
            t = vstart + d + 0.45
            last_spk = 'SYSTEM'
    flush(t)
    t += 2.5
    tl['duration'] = round(t, 3)
    # panels are drawn in the same place, so each one clears before the next arrives
    for a, b in zip(tl['system'], tl['system'][1:]):
        a['end'] = round(max(a['start'] + 1.0, min(a['end'], b['start'] - 0.45)), 3)
    for p in tl['system']:
        p['end'] = round(min(p['end'], tl['duration'] - 0.5), 3)
    # close shots
    for i, s in enumerate(tl['shots']):
        s['end'] = tl['shots'][i + 1]['start'] if i + 1 < len(tl['shots']) else tl['duration']
    return tl


def mix(tl, out_wav):
    n = int(tl['duration'] * SR) + SR
    voice = np.zeros((2, n), np.float32)
    for v in tl['voice']:
        x = load(os.path.join(ROOT, v['path']))
        i = int(v['start'] * SR)
        m = min(x.shape[1], n - i)
        voice[:, i:i + m] += x[:, :m]

    # speech envelope for ducking (attack 0.2 s, release 0.7 s)
    w = int(0.02 * SR)
    frames = np.abs(voice[0, :n // w * w]).reshape(-1, w).max(axis=1)
    active = (frames > 10 ** (-38 / 20)).astype(np.float32)
    env = np.zeros_like(active)
    a, r = 1 - np.exp(-1 / (0.2 / 0.02)), 1 - np.exp(-1 / (0.7 / 0.02))
    # look-ahead so the music dips before the voice lands
    look = int(0.25 / 0.02)
    act = np.concatenate([active[look:], np.zeros(look, np.float32)])
    act = np.maximum(act, active)
    for k in range(1, len(act)):
        c = a if act[k] > env[k - 1] else r
        env[k] = env[k - 1] + c * (act[k] - env[k - 1])
    duck = np.repeat(env, w)
    duck = np.pad(duck, (0, n - len(duck)), constant_values=0)

    # music
    music = np.zeros((2, n), np.float32)
    ev = sorted(tl['music'], key=lambda e: e['time'])
    segs = []
    cur = None
    for e in ev:
        if e['cue'] in STINGS:
            segs.append((e['time'], None, e['cue'], e['gain'], 0.02, 3.0, True))
            continue
        if cur:
            segs.append((cur[0], e['time'], cur[1], cur[2], 2.0, 3.0, False))
            cur = None
        if e['cue'] != 'stop':
            cur = (e['time'], e['cue'], e['gain'])
    if cur:
        segs.append((cur[0], tl['duration'], cur[1], cur[2], 2.0, 3.5, False))
    for start, end, cue, gain, fin, fout, sting in segs:
        x = load(os.path.join(MUSIC, f'{cue}.wav'))
        lvl = MUSIC_GAP_DB + gain - active_rms_db(x)
        i = int(start * SR)
        if sting:
            m = min(x.shape[1], n - i)
            seg = x[:, :m] * env_segment(m, fin, fout)
            music[:, i:i + m] += seg * 10 ** ((lvl + 3) / 20)
            continue
        length = int((end - start + fout) * SR)
        m = min(length, n - i)
        seg = looped(x, m) * env_segment(m, fin, fout)
        music[:, i:i + m] += seg * 10 ** (lvl / 20)
    music *= 10 ** (MUSIC_DUCK_DB * duck / 20)

    # ambience
    amb = np.zeros((2, n), np.float32)
    layers = []
    events = sorted(tl['amb'], key=lambda e: e['time'])
    for e in events:
        if not e['layer']:
            for L in layers:
                if L[1] is None:
                    L[1] = e['time']
        if e['name'] != 'stop':
            layers.append([e['time'], None, e['name'], e['gain']])
    for start, end, name, gain in layers:
        end = end if end is not None else tl['duration']
        x = load(os.path.join(SFX, f'{name}.wav'))
        lvl = AMB_DB + gain - active_rms_db(x)
        i = int(start * SR)
        m = min(int((end - start + 2.5) * SR), n - i)
        amb[:, i:i + m] += looped(x, m, 2.0) * env_segment(m, 2.0, 2.5) * 10 ** (lvl / 20)
    amb *= 10 ** (AMB_DUCK_DB * duck / 20)

    # one-shots
    fx = np.zeros((2, n), np.float32)
    for e in tl['sfx']:
        x = load(os.path.join(SFX, f"{e['name']}.wav"))
        lvl = SFX_DB + e['gain'] + SFX_TRIM.get(e['name'], 0.0) - active_rms_db(x, floor=-50)
        i = max(0, int(e['time'] * SR))
        m = min(x.shape[1], n - i)
        fx[:, i:i + m] += x[:, :m] * 10 ** (lvl / 20)

    master = voice + music + amb + fx
    meter = pyloudnorm.Meter(SR)
    loud = meter.integrated_loudness(master.T)
    master *= 10 ** ((TARGET_LUFS - loud) / 20)
    master = peak_limit(master, CEILING_DB)
    master = master[:, :int(tl['duration'] * SR)]
    sf.write(out_wav, master.T, SR, subtype='PCM_16')
    final = meter.integrated_loudness(master.T)
    return {'lufs': round(float(final), 2), 'peak_db': round(float(20 * np.log10(np.abs(master).max() + 1e-9)), 2),
            'seconds': round(float(master.shape[1] / SR), 2)}


def srt(tl, path):
    def ts(x):
        h, r = divmod(x, 3600)
        m, s = divmod(r, 60)
        return f'{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s - int(s)) * 1000):03d}'
    out, k = [], 1
    for v in tl['voice']:
        if not v['sub']:
            continue
        text = v['text']
        words = text.split()
        chunks, cur = [], []
        for w in words:
            cur.append(w)
            if len(' '.join(cur)) > 84:
                chunks.append(' '.join(cur))
                cur = []
        if cur:
            chunks.append(' '.join(cur))
        total = sum(len(c) for c in chunks)
        t0 = v['start']
        for c in chunks:
            d = v['dur'] * len(c) / total
            lines = c if len(c) <= 42 else c[:c.rfind(' ', 0, len(c) // 2 + 10)] + '\n' + c[c.rfind(' ', 0, len(c) // 2 + 10) + 1:]
            out.append(f'{k}\n{ts(t0)} --> {ts(t0 + d - 0.02)}\n{lines}\n')
            k += 1
            t0 += d
    with open(path, 'w') as f:
        f.write('\n'.join(out))


def main(script):
    ch = parse(script)
    work = os.path.join(ROOT, 'work', f'ch{ch.number:02d}')
    os.makedirs(work, exist_ok=True)
    os.makedirs(os.path.join(ROOT, 'out', 'subs'), exist_ok=True)
    tl = build_timeline(ch)
    stats = mix(tl, os.path.join(work, 'mix.wav'))
    tl['mix'] = stats
    with open(os.path.join(work, 'timeline.json'), 'w') as f:
        json.dump(tl, f, indent=1)
    srt(tl, os.path.join(ROOT, 'out', 'subs', f'ch{ch.number:02d}.srt'))
    print(json.dumps({'chapter': ch.number, 'duration_min': round(tl['duration'] / 60, 2), **stats,
                      'shots': len(tl['shots']), 'voice_lines': len(tl['voice'])}))


if __name__ == '__main__':
    main(sys.argv[1])
