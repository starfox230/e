# Script markup

One file per chapter: `script/chNN.txt`. The renderer builds the timeline strictly from these files and from the measured length of every rendered voice line, so audio and pictures cannot drift apart.

## Headers
```
# 1 | Two Deaths and a Chair          chapter number and title (title card)
@movement I | The Founding        movement card (only on the first chapter of a movement)
```

## Spoken lines (one paragraph per line)
```
N: Narration.
WESKER: Dialogue.            any cast tag from story/bible.md §4
ADOLF: Interior voice.
```
A line may start with `(soft)`, `(cold)`, `(shout)`, `(whisper)`, `(slow)` or `(fast)`. These are delivery hints that adjust speed and processing.

## Cues (they fire at the start of the next spoken line, plus an optional offset in seconds)
```
[img] prompt                  new shot; holds until the next [img]
[loc] Wilmington, Delaware — 14 September 2026        typed lower-third caption
[music] m_rain_piano          crossfade to a music cue (`[music] stop` fades out)
[music] m_tension -6          optional gain in dB
[amb] amb_rain_heavy          start a looped ambience (`[amb] stop`; `[amb+] name` layers a second loop)
[sfx] sfx_thunder +1.5        one-shot, 1.5 s after the next line starts (negative = before)
[pause 1.2]                   silence of that many seconds
[system] TITLE                on-screen System panel, read by the SYSTEM voice
Display line
Label: value                  lines shaped "Label: value" render as aligned rows
[say] Spoken version (optional; default is the title)
[/system]
```
