"""Small DSP toolkit shared by the SFX and music builders. Everything is 48 kHz float32."""
import numpy as np
import soundfile as sf
from scipy import signal
from pedalboard import Pedalboard, Reverb, LowpassFilter, HighpassFilter, Compressor, Limiter, Gain

SR = 48000
RNG = np.random.default_rng(1945)


def t(sec):
    return np.arange(int(sec * SR)) / SR


def white(sec, rng=RNG):
    return rng.standard_normal(int(sec * SR)).astype(np.float32)


def pink(sec, rng=RNG):
    n = int(sec * SR)
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1]
    X /= np.sqrt(f)
    y = np.fft.irfft(X, n)
    return (y / (np.abs(y).max() + 1e-9)).astype(np.float32)


def brown(sec, rng=RNG):
    x = np.cumsum(rng.standard_normal(int(sec * SR)))
    x = signal.sosfilt(signal.butter(1, 20, 'hp', fs=SR, output='sos'), x)
    return (x / (np.abs(x).max() + 1e-9)).astype(np.float32)


def lp(x, fc, order=4):
    return signal.sosfilt(signal.butter(order, fc, 'lp', fs=SR, output='sos'), x, axis=-1).astype(np.float32)


def hp(x, fc, order=4):
    return signal.sosfilt(signal.butter(order, fc, 'hp', fs=SR, output='sos'), x, axis=-1).astype(np.float32)


def bp(x, lo, hi, order=3):
    return signal.sosfilt(signal.butter(order, [lo, hi], 'bp', fs=SR, output='sos'), x, axis=-1).astype(np.float32)


def expdecay(sec, tau):
    return np.exp(-t(sec) / tau).astype(np.float32)


def adsr(n, a, d, s, r):
    """a,d,r in seconds; s level. Sustain fills the remainder."""
    a_n, d_n, r_n = int(a * SR), int(d * SR), int(r * SR)
    s_n = max(0, n - a_n - d_n - r_n)
    env = np.concatenate([
        np.linspace(0, 1, a_n, endpoint=False),
        np.linspace(1, s, d_n, endpoint=False),
        np.full(s_n, s),
        np.linspace(s, 0, r_n),
    ])
    env = np.pad(env, (0, max(0, n - len(env))))[:n]
    return env.astype(np.float32)


def sine_sweep(sec, f0, f1, curve='exp'):
    tt = t(sec)
    if curve == 'exp':
        f = f0 * (f1 / f0) ** (tt / sec)
    else:
        f = f0 + (f1 - f0) * tt / sec
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph).astype(np.float32)


def fm_bell(freq, sec, index=3.0, ratio=1.4, tau=0.8):
    tt = t(sec)
    env = np.exp(-tt / tau)
    mod = index * np.exp(-tt / (tau * 0.4)) * np.sin(2 * np.pi * freq * ratio * tt)
    return (env * np.sin(2 * np.pi * freq * tt + mod)).astype(np.float32)


def place(buf, x, at_sec, gain=1.0):
    """Mix mono/stereo x into buf (stereo, shape [2, n]) at time at_sec."""
    i = int(at_sec * SR)
    if x.ndim == 1:
        x = np.stack([x, x])
    n = min(x.shape[1], buf.shape[1] - i)
    if n > 0 and i >= 0:
        buf[:, i:i + n] += gain * x[:, :n]
    return buf


def stereo(x, width=0.0, delay_ms=0.0):
    if x.ndim == 2:
        return x
    if delay_ms > 0:
        d = int(delay_ms / 1000 * SR)
        r = np.concatenate([np.zeros(d, np.float32), x[:-d] if d else x])
        return np.stack([x, (1 - width) * x + width * r])
    return np.stack([x, x])


def pan(x, p):
    """p in [-1, 1]. Equal-power pan of a mono signal."""
    a = (p + 1) * np.pi / 4
    return np.stack([np.cos(a) * x, np.sin(a) * x]).astype(np.float32)


def fx(x, *plugins):
    board = Pedalboard(list(plugins))
    if x.ndim == 1:
        x = np.stack([x, x])
    return board(x.astype(np.float32), SR)


def reverb(x, room=0.8, wet=0.3, dry=0.8, damp=0.5, width=1.0, tail=0.0):
    if x.ndim == 1:
        x = np.stack([x, x])
    if tail > 0:
        x = np.concatenate([x, np.zeros((2, int(tail * SR)), np.float32)], axis=1)
    return fx(x, Reverb(room_size=room, wet_level=wet, dry_level=dry, damping=damp, width=width))


def normalize(x, peak_db=-1.0):
    p = np.abs(x).max() + 1e-9
    return (x * (10 ** (peak_db / 20) / p)).astype(np.float32)


def fade(x, fin=0.01, fout=0.05):
    x = x.copy()
    n = x.shape[-1]
    a, b = int(fin * SR), int(fout * SR)
    if a:
        x[..., :a] *= np.linspace(0, 1, a)
    if b:
        x[..., n - b:] *= np.linspace(1, 0, b)
    return x


def trim_silence(x, thresh_db=-50):
    mono = np.abs(x).max(axis=0) if x.ndim == 2 else np.abs(x)
    thr = 10 ** (thresh_db / 20)
    idx = np.where(mono > thr)[0]
    if len(idx) == 0:
        return x
    return x[..., idx[0]:idx[-1] + 1]


def load(path, sr=SR):
    y, s = sf.read(path, always_2d=True, dtype='float32')
    y = y.T
    if y.shape[0] == 1:
        y = np.concatenate([y, y])
    if s != sr:
        g = np.gcd(s, sr)
        y = signal.resample_poly(y, sr // g, s // g, axis=1).astype(np.float32)
    return y


def save(path, x):
    if x.ndim == 1:
        x = np.stack([x, x])
    sf.write(path, x.T, SR, subtype='PCM_24')


def make_loop(x, xfade=3.0):
    """Turn a stereo clip into a seamless loop by crossfading its tail into its head."""
    n = int(xfade * SR)
    head, body, tail = x[:, :n], x[:, n:-n], x[:, -n:]
    ramp = np.linspace(0, 1, n, dtype=np.float32)
    joined = tail * (1 - ramp) + head * ramp
    return np.concatenate([body, joined], axis=1)


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)


def mixp(*xs):
    """Sum mono or stereo arrays of different lengths, zero-padding to the longest."""
    xs = [np.stack([x, x]) if x.ndim == 1 else x for x in xs]
    n = max(x.shape[1] for x in xs)
    out = np.zeros((2, n), np.float32)
    for x in xs:
        out[:, :x.shape[1]] += x
    return out if any(True for _ in xs) else out
