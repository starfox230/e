"""Speech QA: transcribe every rendered voice line with Whisper (sherpa-onnx) and flag mismatches.

Usage: python3 pipeline/qa_asr.py 1 [--all]
Writes work/chNN/asr_report.json and prints lines whose word error rate exceeds the threshold.
"""
import os
import re
import sys
import json
import numpy as np
import soundfile as sf
from scipy import signal
import sherpa_onnx
from voice import normalize_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = '/opt/models/asr/sherpa-onnx-whisper-small.en'
THRESH = 0.25


def recognizer():
    return sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=f'{M}/small.en-encoder.int8.onnx', decoder=f'{M}/small.en-decoder.int8.onnx',
        tokens=f'{M}/small.en-tokens.txt', num_threads=4, language='en', task='transcribe')


def words(s):
    s = normalize_text(s).lower().replace('-', ' ')
    s = re.sub(r"[^a-z0-9' ]", ' ', s)
    return [w.strip("'") for w in s.split() if w.strip("'")]


def wer(ref, hyp):
    d = np.zeros((len(ref) + 1, len(hyp) + 1), int)
    d[:, 0] = np.arange(len(ref) + 1)
    d[0, :] = np.arange(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    return d[-1, -1] / max(1, len(ref))


def main(ch):
    work = os.path.join(ROOT, 'work', f'ch{ch:02d}')
    tl = json.load(open(os.path.join(work, 'timeline.json')))
    rec = recognizer()
    report = []
    for v in tl['voice']:
        x, sr = sf.read(os.path.join(ROOT, v['path']), dtype='float32')
        x = x.mean(axis=1) if x.ndim == 2 else x
        x = signal.resample_poly(x, 1, 3).astype(np.float32)  # 48k -> 16k
        # whisper handles <= 30 s windows; split long clips on quiet points
        chunks, n, step = [], len(x), 16000 * 25
        s = 0
        while s < n:
            e = min(n, s + step)
            if e < n:
                win = np.abs(x[e - 16000 * 3:e])
                k = np.convolve(win, np.ones(800) / 800, 'same').argmin()
                e = e - 16000 * 3 + k
            chunks.append(x[s:e])
            s = e
        hyp = []
        for c in chunks:
            st = rec.create_stream()
            st.accept_waveform(16000, c)
            rec.decode_stream(st)
            hyp.append(st.result.text)
        text = ' '.join(hyp).strip()
        w = wer(words(v['text']), words(text))
        report.append({'start': v['start'], 'speaker': v['speaker'], 'wer': round(w, 3), 'text': v['text'], 'heard': text,
                       'path': v['path']})
        if w > THRESH:
            print(f"[{v['start']:7.1f}s] {v['speaker']:8s} WER {w:.2f}\n   REF: {v['text']}\n   ASR: {text}", flush=True)
    json.dump(report, open(os.path.join(work, 'asr_report.json'), 'w'), indent=1)
    ws = [r['wer'] for r in report]
    print(f'ch{ch:02d}: {len(ws)} lines, mean WER {np.mean(ws):.3f}, flagged {sum(w > THRESH for w in ws)}')


if __name__ == '__main__':
    main(int(sys.argv[1]))
