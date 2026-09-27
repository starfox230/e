"""Put a short narrator cue in front of each turn of the inner voice.

In audio the inner voice otherwise arrives out of nowhere: 120 of its 210 turns follow a
Wesker line directly, and with no narrator between them a listener hears one man's voice
jump to another's with no signal that the second one lives in the first one's head. Each
turn now gets a brief cue from the narrator, varied so it never becomes a tic.

  * A turn the narrator already introduces (the line just before it names the voice) is left.
  * The cue goes after any pause before the turn, so the beat stays where it was, and before
    any picture or sound cue, so those still land on the inner voice's own line.
  * Quiet deliveries -- (soft), (whisper), (slow) -- get quieter wording; (cold) gets colder.
  * The wording cycles through a shuffled list per chapter, so no cue repeats back to back.

Every cue names the voice, so running this again finds nothing left to do.

Usage: python3 pipeline/cue_inner_voice.py [--dry-run] [chNN ...]
"""
import os
import re
import sys
import glob
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTRO = re.compile(r'\b(voice|Adolf|Hitler|other one|other half|inside him)\b', re.I)

GENERAL = [
    'The other voice cut in.',
    'The voice inside him answered.',
    "Adolf's voice came from somewhere behind his eyes.",
    'Then the other one, from inside.',
    'The second voice broke in.',
    'Adolf spoke up.',
    'The old voice stirred inside him.',
    "The voice in his head wasn't finished.",
    'Adolf chimed in.',
    'The other half of him answered.',
    'From inside, the other voice.',
    'The inner voice interrupted.',
    "Adolf's voice rose in his head.",
    'The other one spoke, unasked.',
    'The voice behind his eyes had something to add.',
    'Adolf answered from inside.',
    'The second voice came back.',
    'The other voice weighed in.',
]
QUIET = [
    'The other voice, quietly.',
    "Adolf's voice came back, low.",
    'The voice inside him, barely a murmur.',
    'The old voice, slowly.',
    'Adolf, softly, from inside.',
    'The inner voice, almost to itself.',
]
COLD = [
    'The other voice, cold.',
    "Adolf's voice cut in, flat and hard.",
    'The voice inside him, without warmth.',
]


def pools(chapter):
    r = random.Random(chapter)
    out = {}
    for name, words in (('general', GENERAL), ('quiet', QUIET), ('cold', COLD)):
        w = list(words)
        r.shuffle(w)
        out[name] = w
    return out


def cue_chapter(path, dry=False):
    chapter = int(os.path.basename(path)[2:4])
    lines = open(path).read().split('\n')
    pool = pools(chapter)
    used = {k: 0 for k in pool}
    last_cue = None
    out, added = [], []
    last_spk, last_n = None, ''
    for i, line in enumerate(lines):
        m = re.match(r'^([A-Z_]+): (.*)', line)
        if m and m.group(1) == 'ADOLF' and last_spk != 'ADOLF' \
                and not (last_spk == 'N' and INTRO.search(last_n)):
            hint = re.match(r'^\((\w+)\)', m.group(2))
            kind = 'general'
            if hint and hint.group(1) in ('soft', 'whisper', 'slow'):
                kind = 'quiet'
            elif hint and hint.group(1) == 'cold':
                kind = 'cold'
            words = pool[kind]
            cue = words[used[kind] % len(words)]
            if cue == last_cue:                      # never the same cue twice running
                used[kind] += 1
                cue = words[used[kind] % len(words)]
            used[kind] += 1
            last_cue = cue
            # walk back over the cue lines directly above; keep pauses before the new line
            j = len(out)
            while j > 0 and out[j - 1].startswith('[') and not out[j - 1].startswith('[/system'):
                j -= 1
            k = j
            while k < len(out) and out[k].startswith('[pause'):
                k += 1
            out.insert(k, f'N: {cue}')
            added.append((i + 1, cue, line[:80]))
            last_spk, last_n = 'N', cue
        out.append(line)
        if m:
            last_spk = m.group(1)
            if last_spk == 'N':
                last_n = m.group(2)
    if added and not dry:
        open(path, 'w').write('\n'.join(out))
    return added


def main():
    dry = '--dry-run' in sys.argv
    want = [a for a in sys.argv[1:] if a.startswith('ch')]
    paths = sorted(glob.glob(os.path.join(ROOT, 'script', 'ch*.txt')))
    if want:
        paths = [p for p in paths if os.path.basename(p)[:4] in want]
    total = 0
    for p in paths:
        added = cue_chapter(p, dry)
        total += len(added)
        if added and ('-v' in sys.argv or dry):
            for ln, cue, adolf in added[:3]:
                print(f'  {os.path.basename(p)}:{ln}  N: {cue}  ->  {adolf}')
    print(f'{total} cues {"would be " if dry else ""}added')


if __name__ == '__main__':
    main()
