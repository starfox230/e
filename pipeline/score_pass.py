"""Give every scene its own music bed, chosen from what the scene is about.

The complaint this fixes: a cue set at the top of a chapter runs for seven minutes, through
three scene changes, so a tension bed ends up under a quiet conversation and a triumph bed
under a funeral. The scripts mark scenes clearly -- a blank line, then [music]/[img]/[loc] --
so any scene that opens without a [music] line inherits whatever was playing.

This finds those scenes and writes a cue for each, picked by scoring the scene's own narration
against a keyword table. It is deliberately conservative: it never changes a cue that was
written by hand, and a scene whose text scores nothing gets the quiet bed rather than a guess.

Usage: python3 pipeline/score_pass.py [--write] [chNN ...]
"""
import os
import re
import sys
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# cue -> (weight, words). Longer, more specific words are worth more than atmosphere words.
TABLE = {
    'm_system':    [(3, r'\b(panel|System|Points|Shop|Gacha|purchase|balance|tier|roster)\b')],
    'm_horror':    [(3, r'\b(Tyrant|cryogenic|cradle|containment|organism|hive|Myriad|swarm|corpse|plague)\b'),
                    (2, r'\b(virus|pathogen|outbreak|monster|grey|frost)\b')],
    'm_action':    [(3, r'\b(boarding|fired|assault|breach|engagement|task force|marines|landed|gunfire)\b'),
                    (2, r'\b(operation|soldiers|weapons|garrison|fleet)\b')],
    'm_march':     [(3, r'\b(formation|parade|regiment|division|ranks|barracks)\b')],
    'm_sermon':    [(3, r'\b(stadium|crowd|chant|auditorium|keynote|stage|applause|address(?:ed)?)\b'),
                    (2, r'\b(speech|hall|delegates|assembly)\b')],
    'm_corporate': [(3, r'\b(revenue|board|shareholders|acquisition|contract|market|valuation|invoice|plant|quarter)\b'),
                    (2, r'\b(company|price|licence|licensed|accounts|budget)\b')],
    'm_scheme':    [(3, r'\b(dossier|intelligence|negotiat|bargain|leverage|asset network|manoeuvre)\b'),
                    (2, r'\b(plan|arrangement|quietly|privately|offer)\b')],
    'm_tension':   [(3, r'\b(subpoena|inquiry|investigat|hostile|threat|crisis|hostage|ultimatum)\b'),
                    (2, r'\b(committee|hearing|questions|pressure|deadline)\b')],
    'm_sad':       [(4, r'\b(died|death|funeral|grave|buried|cemetery|mourn|widow)\b')],
    'm_triumph':   [(3, r'\b(complete|achieved|record|unanimous|won|victory|anniversary)\b')],
    'm_alien':     [(4, r'\b(Compact|Veth|Choir|Tollhouse|broker|Hegemony|xeno|species in nine hundred)\b')],
    'm_space':     [(3, r'\b(orbit|orbital|vacuum|hull|shipyard|Dock One|light years|colony|system)\b')],
    'm_bell':      [(4, r'\b(appeared|placed|finished existing|materiali|overnight|was not there)\b')],
    'm_hi':        [(3, r'\b(announce|release|reveal|data|results|trial|breakthrough|first time)\b')],
    'm_dawn':      [(3, r'\b(dawn|morning light|sunrise|terrace|river at|window at)\b')],
    'm_lo':        [],          # the default
}
GAINS = {'m_system': -8, 'm_horror': -9, 'm_action': -8, 'm_march': -8, 'm_sermon': -8,
         'm_corporate': -9, 'm_scheme': -9, 'm_tension': -9, 'm_sad': -9, 'm_triumph': -7,
         'm_alien': -8, 'm_space': -8, 'm_bell': -6, 'm_hi': -7, 'm_dawn': -9, 'm_lo': -10}


def scenes(lines):
    """Index ranges of each scene. A scene starts after a blank line at a cue block."""
    starts = [0]
    for i, l in enumerate(lines):
        if l.strip() == '' and i + 1 < len(lines) and lines[i + 1].startswith(('[music]', '[img]', '[loc]')):
            starts.append(i + 1)
    return [(s, (starts[k + 1] if k + 1 < len(starts) else len(lines))) for k, s in enumerate(starts)]


def pick(text):
    best, score = 'm_lo', 0
    for cue, pats in TABLE.items():
        s = sum(w * len(re.findall(p, text, re.I)) for w, p in pats)
        if s > score:
            best, score = cue, s
    return best, score


def process(path, write):
    lines = open(path).read().split('\n')
    out = list(lines)
    added = []
    for a, b in scenes(lines):
        block = lines[a:b]
        if any(l.startswith('[music]') for l in block[:6]):
            continue                       # hand-written cue; leave it alone
        text = ' '.join(l for l in block if re.match(r'^[A-Z]+:', l))
        if len(text.split()) < 40:
            continue                       # too short to be worth a cue change
        cue, score = pick(text)
        if score < 3:
            cue = 'm_lo'
        # insert after the scene's opening [music]-less cue lines, i.e. at the very top
        added.append((a, cue, score))
    for a, cue, score in reversed(added):
        out.insert(a, f'[music] {cue} {GAINS[cue]}')
    if write and added:
        open(path, 'w').write('\n'.join(out))
    return added


def main():
    write = '--write' in sys.argv
    names = [a for a in sys.argv[1:] if not a.startswith('--')]
    paths = ([os.path.join(ROOT, 'script', f'{n}.txt') for n in names]
             if names else sorted(glob.glob(os.path.join(ROOT, 'script', 'ch*.txt'))))
    total = 0
    for p in paths:
        added = process(p, write)
        total += len(added)
        if added:
            print(f'{os.path.basename(p)}: ' + ', '.join(f'{c}({s})' for _, c, s in added))
    print(f'{total} scene beds {"written" if write else "proposed"}')


if __name__ == '__main__':
    main()
