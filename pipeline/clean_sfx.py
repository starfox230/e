"""Take the clicks and the DC offset out of the effects library.

An audit of the 95 rendered effects found two defects in most of them, and both are audible:

  * The waveform starts at full amplitude. A door slam begins at 0.86 of full scale on its
    first sample, which is a click before the sound, and a listener hears it as a fault in
    the recording rather than as part of the effect.
  * A DC offset of up to 0.019, from unbalanced noise sources and asymmetric envelopes. It
    wastes headroom, thumps when the effect starts and ends, and makes the limiter behave
    oddly in the mix.

Both are fixed after the fact rather than in sfx_lib, because the synthesis is fine and it is
the edges that are wrong. One-shots get a 3 ms raised-cosine fade in and 8 ms out, which is
short enough to leave a transient intact -- a gunshot still cracks -- and long enough that no
sample boundary is a step. Ambiences get the DC removed and nothing else: they are looped, the
loop seam was crossfaded when they were built, and fading the file edges would put a dip in
the middle of a continuous background every time it wrapped.

Usage: python3 pipeline/clean_sfx.py [--check]
"""
import os
import sys
import glob
import numpy as np
import soundfile as sf
from scipy import signal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SFX = os.path.join(ROOT, 'assets', 'sfx')
FADE_IN = 0.003
FADE_OUT = 0.008
PEAK = 10 ** (-1.0 / 20)


def dc_block(x, sr):
    x = x - x.mean(axis=0, keepdims=True)
    b, a = signal.butter(2, 18.0 / (sr / 2), btype='high')
    return signal.filtfilt(b, a, x, axis=0).astype('float32')


def edges(x, sr):
    n_in, n_out = int(FADE_IN * sr), int(FADE_OUT * sr)
    if len(x) > n_in + n_out:
        x[:n_in] *= (0.5 - 0.5 * np.cos(np.pi * np.linspace(0, 1, n_in)))[:, None]
        x[-n_out:] *= (0.5 + 0.5 * np.cos(np.pi * np.linspace(0, 1, n_out)))[:, None]
    return x


def main():
    check = '--check' in sys.argv
    fixed = 0
    for p in sorted(glob.glob(os.path.join(SFX, '*.wav'))):
        name = os.path.basename(p)
        x, sr = sf.read(p, dtype='float32', always_2d=True)
        before_dc = float(np.abs(x.mean()))
        before_head = float(np.abs(x[:int(0.002 * sr)]).max())
        y = dc_block(x, sr)
        if not name.startswith('amb_'):
            y = edges(y, sr)
            peak = np.abs(y).max()
            if peak > 0:
                y = y * min(PEAK / peak, 1.0) if peak > PEAK else y
        else:
            peak = np.abs(y).max()
            if peak > PEAK:
                y = y * (PEAK / peak)
        after_head = float(np.abs(y[:int(0.002 * sr)]).max())
        if check:
            if before_dc > 0.002 or before_head > 0.05:
                print(f'{name:30s} dc {before_dc:.4f} -> {float(np.abs(y.mean())):.4f}   '
                      f'head {before_head:.2f} -> {after_head:.2f}')
            continue
        sf.write(p, y, sr)
        fixed += 1
    if not check:
        print(f'{fixed} effects rewritten')


if __name__ == '__main__':
    main()
