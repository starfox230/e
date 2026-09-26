"""Parse a chapter script (see script/FORMAT.md) into a flat list of events."""
import re
from dataclasses import dataclass, field

HINTS = ('soft', 'cold', 'shout', 'whisper', 'slow', 'fast', 'phone')


@dataclass
class Ev:
    kind: str                 # line | img | loc | music | amb | amb+ | sfx | pause | system
    speaker: str = ''
    text: str = ''
    hint: str = ''
    name: str = ''
    offset: float = 0.0
    gain: float = 0.0
    dur: float = 0.0
    lines: list = field(default_factory=list)
    say: str = ''
    lineno: int = 0


@dataclass
class Chapter:
    number: int
    title: str
    movement: tuple = None     # (numeral, title) or None
    events: list = field(default_factory=list)


def _cue_args(rest):
    """'name +1.5 g-6' -> (name, offset, gain)."""
    parts = rest.split()
    name, off, gain = (parts[0] if parts else ''), 0.0, 0.0
    for p in parts[1:]:
        if p.startswith('g'):
            gain = float(p[1:])
        elif re.fullmatch(r'[+-]?\d+(\.\d+)?', p):
            # a bare number after a music cue is gain in dB, after sfx it is an offset in seconds
            off = float(p)
    return name, off, gain


def parse(path):
    ch = None
    events = []
    sys_block = None
    with open(path, encoding='utf-8') as f:
        raw = f.read().splitlines()
    for i, line in enumerate(raw, 1):
        s = line.strip()
        if sys_block is not None:
            if s == '[/system]':
                events.append(sys_block)
                sys_block = None
            elif s.startswith('[say]'):
                sys_block.say = s[5:].strip()
            elif s:
                sys_block.lines.append(s)
            continue
        if not s:
            continue
        if s.startswith('# '):
            m = re.match(r'#\s*(\d+)\s*\|\s*(.+)', s)
            ch = Chapter(int(m.group(1)), m.group(2).strip())
            continue
        if s.startswith('@movement'):
            num, title = [x.strip() for x in s[len('@movement'):].split('|', 1)]
            ch.movement = (num, title)
            continue
        m = re.match(r'\[(\w+\+?|pause [\d.]+)\]\s*(.*)', s)
        if m:
            tag, rest = m.group(1), m.group(2).strip()
            if tag.startswith('pause'):
                events.append(Ev('pause', dur=float(tag.split()[1]), lineno=i))
            elif tag in ('title', 'cards'):
                events.append(Ev(tag, lineno=i))
            elif tag == 'img':
                events.append(Ev('img', text=rest, lineno=i))
            elif tag == 'loc':
                events.append(Ev('loc', text=rest, lineno=i))
            elif tag == 'system':
                sys_block = Ev('system', name=rest, lineno=i)
            elif tag in ('music', 'amb', 'amb+'):
                parts = rest.split()
                gain = float(parts[1]) if len(parts) > 1 else 0.0
                events.append(Ev(tag, name=parts[0], gain=gain, lineno=i))
            elif tag == 'sfx':
                name, off, gain = _cue_args(rest)
                events.append(Ev('sfx', name=name, offset=off, gain=gain, lineno=i))
            else:
                raise ValueError(f'{path}:{i}: unknown cue [{tag}]')
            continue
        m = re.match(r'([A-Z][A-Z_]*):\s*(.+)', s)
        if m:
            spk, text = m.group(1), m.group(2).strip()
            hint = ''
            hm = re.match(r'\((\w+)\)\s*(.+)', text)
            if hm and hm.group(1) in HINTS:
                hint, text = hm.group(1), hm.group(2)
            events.append(Ev('line', speaker=spk, text=text, hint=hint, lineno=i))
            continue
        raise ValueError(f'{path}:{i}: cannot parse: {s[:80]}')
    if sys_block is not None:
        raise ValueError(f'{path}: unterminated [system] block')
    ch.events = events
    return ch


if __name__ == '__main__':
    import sys
    c = parse(sys.argv[1])
    from collections import Counter
    print(c.number, c.title, c.movement)
    print(Counter(e.kind for e in c.events))
    print(Counter(e.speaker for e in c.events if e.kind == 'line'))
    words = sum(len(e.text.split()) for e in c.events if e.kind == 'line')
    print('spoken words', words)
