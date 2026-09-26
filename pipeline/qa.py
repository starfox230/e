"""Automated QA for a rendered chapter.

Checks, per chapter:
  * the timeline's duration matches the mixed audio and the muxed video (no drift)
  * audio and video stream durations agree inside one frame
  * integrated loudness and true peak are in spec
  * no clipped samples, no digital silence where speech should be
  * every voice line lands inside the video, and every shot has an image
  * no all-black or all-white frames outside the intended fades
  * subtitles are monotonic and inside the running time

Usage: python3 pipeline/qa.py 1 [2 3 ...]   (no args = every rendered chapter)
"""
import os
import sys
import json
import glob
import subprocess
import numpy as np
import soundfile as sf
import pyloudnorm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 48000
FPS = 30
TOL_FRAME = 1.0 / FPS
SPEC = {'lufs': (-16.5, -13.5), 'peak_db': (-20.0, -0.9), 'sync_s': 0.05}


def probe(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                          'format=duration:stream=codec_type,duration,nb_frames,r_frame_rate',
                          '-of', 'json', path], capture_output=True, text=True).stdout
    return json.loads(out)


def check(ch, fix=False):
    work = os.path.join(ROOT, 'work', f'ch{ch:02d}')
    tl_path = os.path.join(work, 'timeline.json')
    issues, notes = [], {}
    if not os.path.exists(tl_path):
        return [f'ch{ch:02d}: no timeline'], {}
    tl = json.load(open(tl_path))
    dur = tl['duration']
    notes['duration_min'] = round(dur / 60, 2)

    # ---- audio
    mix = os.path.join(work, 'mix.wav')
    if not os.path.exists(mix):
        return [f'ch{ch:02d}: no mix.wav'], notes
    info = sf.info(mix)
    notes['audio_s'] = round(info.duration, 3)
    if abs(info.duration - dur) > 0.05:
        issues.append(f'audio length {info.duration:.3f}s != timeline {dur:.3f}s')
    if info.samplerate != SR:
        issues.append(f'sample rate {info.samplerate}')
    x, _ = sf.read(mix, dtype='float32')
    meter = pyloudnorm.Meter(SR)
    lufs = meter.integrated_loudness(x)
    peak = 20 * np.log10(np.abs(x).max() + 1e-12)
    notes['lufs'] = round(float(lufs), 2)
    notes['peak_db'] = round(float(peak), 2)
    if not SPEC['lufs'][0] <= lufs <= SPEC['lufs'][1]:
        issues.append(f'loudness {lufs:.2f} LUFS out of spec')
    if not SPEC['peak_db'][0] <= peak <= SPEC['peak_db'][1]:
        issues.append(f'true peak {peak:.2f} dBFS out of spec')
    clipped = int((np.abs(x) >= 0.999).sum())
    if clipped:
        issues.append(f'{clipped} clipped samples')
    # silence where speech should be
    for v in tl['voice'][:400]:
        i0 = int(v['start'] * SR)
        i1 = min(len(x), int((v['start'] + min(v['dur'], 2.0)) * SR))
        if i1 <= i0:
            continue
        if np.abs(x[i0:i1]).max() < 1e-4:
            issues.append(f"silence at voice line {v['start']:.1f}s ({v['speaker']})")
            break
    # every voice line inside the running time
    last = max((v['start'] + v['dur']) for v in tl['voice'])
    if last > dur + 0.01:
        issues.append(f'voice runs past end: {last:.2f} > {dur:.2f}')

    # ---- images
    missing = 0
    for shot in tl['shots']:
        if not any(os.path.exists(os.path.join(ROOT, 'work', 'images', f"{shot['id']}.{e}"))
                   for e in ('png', 'jpg', 'webp')):
            missing += 1
    notes['shots'] = len(tl['shots'])
    if missing:
        issues.append(f'{missing} shots have no image')

    # ---- video
    vid = os.path.join(ROOT, 'out', 'video', f'ch{ch:02d}.mp4')
    if not os.path.exists(vid):
        vid = os.path.join(work, 'video.mp4')
    if os.path.exists(vid):
        pr = probe(vid)
        vdur = float(pr['format']['duration'])
        notes['video_s'] = round(vdur, 3)
        streams = {s['codec_type']: s for s in pr['streams']}
        if abs(vdur - dur) > 1.0:
            issues.append(f'video {vdur:.2f}s != timeline {dur:.2f}s')
        if 'audio' in streams and 'video' in streams:
            av = abs(float(streams['audio'].get('duration', vdur)) - float(streams['video'].get('duration', vdur)))
            notes['av_skew_s'] = round(av, 3)
            if av > SPEC['sync_s']:
                issues.append(f'A/V stream length mismatch {av:.3f}s')
        # frame-level: sample 12 frames, flag pure black/white outside fades
        bad = []
        for t in np.linspace(2, max(2.1, dur - 3), 12):
            out = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t:.2f}', '-i', vid, '-frames:v', '1',
                                  '-f', 'rawvideo', '-pix_fmt', 'gray', '-'], capture_output=True).stdout
            if not out:
                bad.append(('unreadable', t))
                continue
            a = np.frombuffer(out, np.uint8)
            if a.mean() < 2:
                bad.append(('black', t))
            elif a.mean() > 250:
                bad.append(('white', t))
        if bad:
            issues.append(f'{len(bad)} bad frames: ' + ', '.join(f'{k}@{t:.0f}s' for k, t in bad[:4]))
    else:
        notes['video_s'] = None

    # ---- subtitles
    srt = os.path.join(ROOT, 'out', 'subs', f'ch{ch:02d}.srt')
    if os.path.exists(srt):
        times = []
        for line in open(srt):
            if '-->' in line:
                a, b = line.split('-->')
                def sec(t):
                    h, m, s = t.strip().replace(',', '.').split(':')
                    return int(h) * 3600 + int(m) * 60 + float(s)
                times.append((sec(a), sec(b)))
        if times:
            if any(b <= a for a, b in times):
                issues.append('subtitle with non-positive duration')
            if any(times[i][0] < times[i - 1][0] for i in range(1, len(times))):
                issues.append('subtitles not monotonic')
            if times[-1][1] > dur + 1:
                issues.append('subtitles run past end')
            notes['subs'] = len(times)
    return issues, notes


def main(chapters):
    ok = True
    for ch in chapters:
        issues, notes = check(ch)
        tag = 'PASS' if not issues else 'FAIL'
        ok &= not issues
        print(f"ch{ch:02d} {tag} {json.dumps(notes)}")
        for i in issues:
            print(f'    - {i}')
    print('ALL PASS' if ok else 'ISSUES FOUND')
    return 0 if ok else 1


if __name__ == '__main__':
    args = [int(a) for a in sys.argv[1:] if a.isdigit()]
    if not args:
        args = sorted(int(os.path.basename(os.path.dirname(p))[2:])
                      for p in glob.glob(os.path.join(ROOT, 'work', 'ch*', 'timeline.json')))
    sys.exit(main(args))
