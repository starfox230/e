"""Generate the film's scene art with SDXL-Turbo, from the [img] prompts in the timelines.

The prompts were written for a human reader and a procedural renderer, so they go through
prepare() before they reach the model. What it deals with, and why:

  * Character tags expand to a look, long form when a shot has one character and short when
    it has several -- CLIP reads 77 tokens and drops the rest without a word, so three long
    descriptions would crowd out the scene they were meant to be in.
  * CLIP has no negation. "without sunglasses" draws sunglasses and "no people" draws
    people, and SDXL-Turbo runs without guidance, so there is no negative prompt to lean on.
    Negations become the positive thing meant: "empty", "alone", "in shadow".
  * Legible text cannot be drawn and garbled text looks broken, so requests for a sign or
    screen that *reads* something keep the sign and lose the words.
  * The company's mark is described as a plain crimson emblem. The word "umbrella" beside
    "red and white" is a description of a real game studio's logo, and a model trained on
    the internet will draw that logo.
  * A consistent style tail goes last, so if anything is truncated it is the style, never
    the subject.

Usage:
  python3 pipeline/gen_images.py --report            print every prompt as the model sees it
  python3 pipeline/gen_images.py --sample ID ...     generate a few shots into the scratch dir
  python3 pipeline/gen_images.py [--limit N]         generate everything, resumably
"""
import os
import re
import sys
import json
import glob
import time
import hashlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get('GEN_OUT') or os.path.join(ROOT, 'work', 'images_gen')
GEN_W, GEN_H = 768, 448          # generation size; cropped to 768x432 (16:9) before upscaling
STYLE = 'cinematic film still, 35mm, dramatic lighting, rich shadows, realistic'

# Short forms for shots with more than one character in them.
SHORT = {
    'WESKER': 'a platinum-blond man in black sunglasses',
    'DANA': 'a grey-haired woman in a cardigan',
    'LESSING': 'a gaunt grey-haired man in wire glasses',
    'EVELYN': 'a young auburn-haired woman in a lab coat',
    'TOMAS': 'a young curly-haired Latino man',
    'NADIA': 'a Black woman with close-cropped hair in a lab coat',
    'WALTER': 'an old white-haired security guard',
    'CATHERINE': 'a Black British woman in a charcoal suit',
    'HALE': 'a tall shaved-head man in black tactical gear',
    'PELL': 'a thin pale man in round spectacles',
    'GUS': 'a stocky maintenance man in navy coveralls',
    'HANNAH': 'a black-haired woman journalist in a trench coat',
    'TOLLIVER': 'a tanned executive in an open-collar shirt',
    'KUCHAR': 'a heavyset man in a baseball cap',
    'DELGADO': 'a Hispanic inspector with a clipboard',
    'CARRICK': 'a tired FBI agent in a dark suit',
    'HADDAD': 'a dark-haired woman epidemiologist',
    'SLOANE': 'a lean silver-haired man in a navy suit',
    'ASHCOMBE': 'a patrician Englishman in a grey suit',
    'WEN': 'a severe Chinese woman in a dark jacket',
    'BELOV': 'a heavy-browed Russian colonel in a greatcoat',
    'HALVORSEN': 'a silver-haired Scandinavian woman in white',
    'WYATT': 'a silver-haired retired general',
    'CRANE': 'a messy-haired tech founder in a black t-shirt',
    'TYRANT': 'a towering grey-skinned giant in a dark trench coat',
    'SILENT': 'security men in identical black suits',
    'UMBRELLA': 'a crimson circular emblem',
}
# Long-form overrides for the character sheet in story/visuals.json.
LONG = {
    # the name adds nothing CLIP can draw except the game's own render of him
    'WESKER': 'a tall man with slicked-back platinum blond hair, black sunglasses, black high-collared long coat',
    'UMBRELLA': 'a crimson circular emblem',
}
# Him with the glasses off is a turning point every time it happens in the story; the
# standard look would put the glasses straight back on.
BARE_EYED = {
    'long': 'a tall man with slicked-back platinum blond hair, pale grey eyes, black high-collared long coat',
    'short': 'a platinum-blond man with pale grey eyes',
}
NO_GLASSES = re.compile(r'without (his )?sunglasses|sunglasses (off|removed|in (his )?hand|folded)|'
                        r'(takes|took|taking|removing|removes|removed) (off )?(his )?sunglasses|eyes visible', re.I)
# the phrases that contradict the glasses are removed; "his eyes visible" is only a cue and stays
GLASSES_PHRASE = re.compile(r'\s*\b(without (his )?sunglasses|sunglasses (off|removed|in (his )?hand|folded)|'
                            r'(takes|took|taking|removing|removes|removed) (off )?(his )?sunglasses)\b', re.I)
# A year before 1950 is what invites period uniforms and insignia into a picture; the
# narration and the location caption already carry the date.
OLD_YEAR = re.compile(r',?\s*\b(1[89][0-4]\d)s?\b')
# The character sheet fixes each age as of 2026 and the story runs for decades, so when a
# prompt gives an age of its own the sheet's is dropped rather than contradicted.
AGED = re.compile(r'\b(old|older|elderly|aged|ageing|aging|grey-haired|white-haired|ageless|'
                  r'decades later|years later|\d+-year-old|in (his|her) (late |early |mid-)?'
                  r'(thirties|forties|fifties|sixties|seventies|eighties|nineties))\b', re.I)

# CLIP draws what is named even when it is negated; say the positive thing instead.
NEGATIONS = [
    (r',?\s*no people\b', ', empty, deserted'),
    (r',?\s*no faces? visible\b', ', deep shadow'),
    (r',?\s*no entourage\b', ', alone'),
    (r',?\s*no one speaking\b', ', utter silence'),
    (r'\bnobody blinking\b', 'every face unblinking'),
    (r',?\s*not glowing\b', ''),
]
# A sign, screen or document that "reads" something: keep the object, lose the words.
TEXT_REQ = re.compile(
    r'\s*,?\s*\b(reading|labelled|labeled|that reads|which reads|saying|marked|stamped|inscribed|'
    r'titled|headlined|captioned|spelling out|with the (words|word|name|headline|title))\s+'
    r'(\"[^\"]*\"|“[^”]*”|\'[^\']*\'|[A-Z0-9][A-Z0-9 \-:&\'’.!?%$/]*[A-Z0-9%!?)])')
MARK = [
    (r'\b(a |an |the )?red[- ]and[- ]white (octagonal )?(umbrella )?(emblem|logo|symbol|insignia|mark|sign)\b',
     'a crimson circular emblem'),
    (r'\b(a |an |the )?(octagonal )?umbrella (emblem|logo|symbol|insignia|mark)\b', 'a crimson circular emblem'),
    (r'\bUmbrella (Life Sciences |Corporation |corporate )?(headquarters|HQ)\b', 'corporate headquarters'),
    # the company's name, as adjective ("the Umbrella tower") and as noun ("awarded to Umbrella")
    (r"\bUmbrella's\b", "the corporation's"),
    (r'\ban Umbrella\b', 'a corporate'),
    (r'\bUmbrella(?= [a-z])', 'corporate'),
    (r'\bthe Umbrella\b', 'the corporation'),
    (r'\bUmbrella\b', 'the corporation'),
    (r'\bred[- ]and[- ]white\b', 'crimson'),
]


# The sheet gives each character's age at their first appearance -- Carrick is 42 when he
# walks on in 2029, and the script has him fifty-eight in 2046. Evelyn stops ageing when she
# becomes the Progenitor Matriarch (ch 42, 2048); the summoned staff carry no age at all.
SHEET_YEAR = 2026
AGELESS_FROM = {'EVELYN': 2048}
_FIRST = {}


def first_seen():
    """Year each character tag first appears in an image prompt."""
    if not _FIRST:
        years = chapter_years()
        for tl_path in sorted(glob.glob(os.path.join(ROOT, 'work', 'ch*', 'timeline.json'))):
            tl = json.load(open(tl_path))
            for sh in tl['shots']:
                for tag in re.findall(r'\{([A-Z_]+)\}', sh['prompt']):
                    _FIRST.setdefault(tag, years[tl['chapter']])
    return _FIRST


def chapter_years():
    """Story year of each chapter, read off its location captions and carried forward."""
    years, yr = {}, SHEET_YEAR
    for tl_path in sorted(glob.glob(os.path.join(ROOT, 'work', 'ch*', 'timeline.json'))):
        tl = json.load(open(tl_path))
        ys = [int(y) for l in tl['loc'] for y in re.findall(r'\b(20\d\d|21\d\d)\b', l['text'])]
        if ys:
            yr = min(ys)
        years[tl['chapter']] = yr
    return years


def age_to(key, desc, year):
    """Re-age a sheet description to the story year: '42-year-old' in 2026 is 62 in 2046."""
    if not year:
        return desc
    m = re.search(r'\b(a|an) (\d+)-year-old\b', desc)
    born = first_seen().get(key, SHEET_YEAR)
    grown = year - born
    if key in AGELESS_FROM:
        grown = min(year, AGELESS_FROM[key]) - born
    if m:
        age = int(m.group(2)) + max(0, grown)
        rep = 'an elderly' if age >= 75 else f'a {age}-year-old'
        desc = desc[:m.start()] + rep + desc[m.end():]
        if age >= 45:
            desc = desc.replace('young ', '')
    return desc


def looks():
    v = json.load(open(os.path.join(ROOT, 'story', 'visuals.json')))
    out = {k: d for k, d in v.items() if not k.startswith('_')}
    out.update(LONG)
    return out


def prepare(prompt, tok=None, look=None, year=None):
    """Turn a script's [img] prompt into what the model is given."""
    look = look or looks()
    p = prompt.strip()
    p = re.sub(r',?\s*\bcinematic( wide shot| close-up)?\s*$', '', p)     # the style tail replaces it
    p = re.sub(r'\b(dark )?cinematic\b,?\s*', '', p).strip(' ,')
    for pat, rep in NEGATIONS:
        p = re.sub(pat, rep, p, flags=re.I)
    # The story's president is fictional. An office named without a face invites the model
    # to supply a real officeholder's, so she is always described.
    if re.search(r'\bpresident\b', p, re.I) and not re.search(r'woman president|her back', p, re.I):
        p = re.sub(r'\b(a|the) (seated )?president\b', r'\1 \2grey-haired woman president', p, flags=re.I)
    p = TEXT_REQ.sub('', p)
    tags = re.findall(r'\{([A-Z_]+)\}', p)
    many = len(set(t for t in tags if t != 'UMBRELLA')) > 1
    bare = bool(NO_GLASSES.search(p))
    aged = bool(AGED.search(p))

    def sub(m):
        k = m.group(1)
        if k == 'WESKER' and bare:
            return BARE_EYED['short' if many else 'long']
        d = SHORT.get(k, k.lower()) if many else look.get(k, SHORT.get(k, k.lower()))
        if aged:
            # the prompt gives an age of its own; the sheet's would contradict it
            d = re.sub(r'\b(a|an) \d+-year-old\b', 'a', d)
        else:
            base = look.get(k, '')
            m = re.search(r'(\d+)-year-old', base)
            if many and m:
                # short forms carry no number, so carry the ageing over as a word
                d = age_to(k, f'a {m.group(1)}-year-old ' + d[2:] if d.startswith('a ') else d, year)
                d = re.sub(r'^(a|an) (\d+)-year-old ', lambda q: 'an older ' if int(q.group(2)) >= 55 else 'a ', d)
            else:
                d = age_to(k, d, year)
        return d
    p = re.sub(r'\{([A-Z_]+)\}', sub, p)
    if bare:
        p = GLASSES_PHRASE.sub('', p)
        p = re.sub(r'\s+,', ',', p)
    p = OLD_YEAR.sub('', p)
    for pat, rep in MARK:
        p = re.sub(pat, rep, p)
    # a run of capitals left over is text the model would try, and fail, to letter
    p = re.sub(r'\b[A-Z]{3,}(\s+[A-Z]{2,})+\b', '', p)
    p = re.sub(r'\s{2,}', ' ', re.sub(r'(,\s*)+', ', ', p)).strip(' ,')
    # dropping an age leaves "a FBI agent", "a auburn-haired woman": mend the article
    p = re.sub(r'\b([Aa]) (?=(?:[aeiou]\w|FBI\b|MI\d|NSA\b|SUV\b|RAF\b|FSB\b|8\d?-|11-|18-))',
               lambda m: m.group(1) + 'n ', p)
    full = f'{p}, {STYLE}'
    if tok is not None and len(tok(full).input_ids) > 77:
        full = f'{p}, cinematic film still'                  # shed style before subject
    return full


def shots():
    """(shot id, prompt, story year) for every shot in the film, first occurrence only."""
    years = chapter_years()
    out, seen = [], set()
    for tl_path in sorted(glob.glob(os.path.join(ROOT, 'work', 'ch*', 'timeline.json'))):
        tl = json.load(open(tl_path))
        for s in tl['shots']:
            if s['id'] not in seen:
                seen.add(s['id'])
                out.append((s['id'], s['prompt'], years[tl['chapter']]))
    return out


def seed_of(key):
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)


def grade(img, seed):
    """The film's finishing pass: a light vignette and fine grain. The grain also hides the
    upscaler's tendency to smooth skin toward an airbrushed look."""
    import numpy as np
    from PIL import Image
    a = np.asarray(img, np.float32) / 255.0
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    a *= (1 - 0.30 * np.clip(d - 0.45, 0, 1) ** 1.6)[..., None]
    rng = np.random.default_rng(seed)
    a += rng.normal(0, 0.014, (h, w, 1)).astype(np.float32)
    return Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))


def main():
    import numpy as np
    from transformers import CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained(os.path.join(ROOT, 'work', 'models', 'sdxl-turbo-ov', 'tokenizer'))
    look = looks()
    all_shots = shots()

    if '--report' in sys.argv:
        over = 0
        for sid, p, yr in all_shots:
            q = prepare(p, tok, look, yr)
            n = len(tok(q).input_ids)
            over += n > 77
            print(f'{sid} [{n:2d}] {q}')
            if '--diff' in sys.argv:
                print(f'{"":14s}  was: {p}')
        print(f'# {len(all_shots)} shots, {over} still over 77 tokens', file=sys.stderr)
        return 0

    import sdxl_ov
    sample = '--sample' in sys.argv
    if sample:
        ids = [a for a in sys.argv[sys.argv.index('--sample') + 1:] if not a.startswith('--')]
        todo = [x for x in all_shots if x[0] in ids]
        out_dir = '/tmp/claude-0/-home-user-e/b62efde1-8f1a-59c0-a070-04b692eaca77/scratchpad/samples'
        variants = ['']
    else:
        todo, out_dir, variants = all_shots, OUT, ['', '_v1']
    os.makedirs(out_dir, exist_ok=True)
    limit = int(sys.argv[sys.argv.index('--limit') + 1]) if '--limit' in sys.argv else None

    jobs = [(sid, p, yr, v) for sid, p, yr in todo for v in variants
            if not os.path.exists(os.path.join(out_dir, f'{sid}{v}.jpg'))]
    done = len(todo) * len(variants) - len(jobs)
    if '--shard' in sys.argv:
        i, n = (int(x) for x in sys.argv[sys.argv.index('--shard') + 1].split('/'))
        jobs = jobs[i::n]
    if limit:
        jobs = jobs[:limit]
    print(f'{len(jobs)} images to generate this run ({done} of {len(todo) * len(variants)} already on disk)', flush=True)
    if not jobs:
        return 0
    t0 = time.time()
    threads = int(sys.argv[sys.argv.index('--threads') + 1]) if '--threads' in sys.argv else None
    gen = sdxl_ov.SDXLTurbo(GEN_W, GEN_H, 1, threads=threads)
    up = sdxl_ov.Upscaler(GEN_W, 432, threads=threads)
    print(f'models ready in {time.time() - t0:.0f}s', flush=True)
    log = open(os.path.join(out_dir, 'prompts.tsv'), 'a')
    t0 = time.time()
    for k, (sid, p, yr, v) in enumerate(jobs):
        q = prepare(p, tok, look, yr)
        s = seed_of(sid + v)
        img = gen(q, s).crop((0, 8, GEN_W, 440))
        img = grade(up(img), s)
        tmp = os.path.join(out_dir, f'.{sid}{v}.jpg')
        img.save(tmp, quality=92, subsampling=1, optimize=True)
        os.replace(tmp, os.path.join(out_dir, f'{sid}{v}.jpg'))   # never leave a half-written frame
        log.write(f'{sid}{v}\t{s}\t{q}\n')
        log.flush()
        if (k + 1) % 10 == 0 or k + 1 == len(jobs):
            el = time.time() - t0
            rate = el / (k + 1)
            print(f'{k + 1}/{len(jobs)}  {rate:.1f}s/img  eta {(len(jobs) - k - 1) * rate / 3600:.2f}h', flush=True)
    return 0


if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
