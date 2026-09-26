# UMBRELLA — a 50-chapter What-If epic, built end to end in this repository

**What if Adolf Hitler was reborn as Albert Wesker, in our world, in September 2026, with a System to build Umbrella?**

A long-form narrated film: ~11½ hours, 50 chapters in 6 movements, with a realistic narrator, a
cast of character voices, scene art that changes every 15–25 seconds, on-screen System panels,
and a sparse sound and music bed in the *Chrysalis* tradition — effects that punctuate, never wallpaper.

Everything here is generated from source in this repository. Nothing is downloaded at render time
except the two model files noted below.

---

## Watch / listen

| File | What it is |
|---|---|
| `out/umbrella_full.mp4` | the whole film, 1080p30, with embedded chapter marks |
| `out/video/chNN.mp4` | one file per chapter |
| `out/umbrella_full.srt` | subtitles for the full film |
| `out/subs/chNN.srt` | subtitles per chapter |
| `out/chapters.txt` | YouTube chapter markers, ready to paste into a description |
| `out/manifest.json` | per-chapter offsets, durations and QA numbers |
| `DESCRIPTION.md` | a ready-to-post title, description and tags |

## Story

| File | What it is |
|---|---|
| `story/bible.md` | tone contract, the man, the System's rules, full cast with voice assignments, factions, timeline |
| `story/storyboard.md` | all 50 chapters, beat by beat, with the thresholds marked |
| `story/ledger.md` | running continuity: Points, levels, stats, purchases, who knows what, and when |
| `story/visuals.json` | character look sheet used to expand `{NAME}` tags in image prompts |
| `script/chNN.txt` | the chapters themselves, in the markup below |
| `script/FORMAT.md` | the markup specification |

### Script markup

```
# 12 | The Black Interior        chapter number and title
@movement II | The Company…      movement card (first chapter of a movement only)

N: Narration.                    any cast tag from the bible
WESKER: Dialogue.
ADOLF: Interior voice.
LESSING: (phone) Filtered line.  delivery hints: soft cold shout whisper slow fast phone

[img] a dark basement, {WESKER} standing at a frosted window, cinematic
[loc] Wilmington, Delaware — 14 September 2026
[music] m_rain_piano -4          crossfade to a cue, optional gain in dB ([music] stop)
[amb] amb_rain_heavy             looped ambience ([amb+] layers a second, [amb] stop)
[sfx] sfx_thunder +1.5           one-shot, offset in seconds
[pause 1.2]
[system] MISSION COMPLETE        on-screen System panel, spoken by the System voice
Reward: 1,500 Points
[say] Mission complete.
[/system]
```

Every time in the finished film comes from the **measured length of each rendered voice line**, so
the picture cannot drift from the sound: the same timeline drives the mix and the frames.

## Pipeline

| Stage | File | What it does |
|---|---|---|
| Parse | `pipeline/parse.py` | script → typed events |
| Voice | `pipeline/voice.py` | Kokoro-82M ONNX, 54 voices + blends, per-character processing chains, number/abbreviation normalisation, content-hash cache |
| Sound | `pipeline/sfx_lib.py` | 90 effects and ambiences: synthesis (`pipeline/dsp.py`) plus recorded sources |
| Crowds | `pipeline/crowd_lib.py` | murmur, cheers, chants and the Creed, built by layering dozens of TTS voices |
| Music | `pipeline/music_lib.py` | 17 cues composed in code as MIDI, rendered with FluidSynth + MuseScore General, mastered |
| Timeline + mix | `pipeline/audio.py` | builds the timeline, ducks music under speech with look-ahead, mixes to −15 LUFS with a look-ahead limiter, writes SRT |
| Scene art | `pipeline/art.py` | procedural cinematic renderer — layered silhouettes, atmospheric depth, window light spill, wet-ground reflection, particles, film grain |
| Graphics | `pipeline/graphics.py` | System panels, location captions, chapter/movement/title cards, the emblem |
| Video | `pipeline/video.py` | frame compositor: sub-pixel Ken Burns, crossfades, panel type-on, caption wipes, vignette, fades |
| Render | `pipeline/render_chapter.py` | splits a chapter across cores, concatenates without re-encoding, muxes audio |
| Assemble | `pipeline/assemble.py` | full film, shifted subtitles, chapter markers, manifest |
| QA | `pipeline/qa.py` | duration match, A/V skew, loudness, true peak, clipping, silence, missing art, black frames, subtitle sanity |
| QA (speech) | `pipeline/qa_asr.py` | transcribes every rendered line with Whisper and flags mismatches by word error rate |
| Cue check | `pipeline/check_cues.py` | fails the build if a script references a sound or cue that does not exist |

### Build

```bash
python3 pipeline/sfx_lib.py          # effects and ambiences  → assets/sfx
python3 pipeline/crowd_lib.py        # crowd voices           → assets/sfx
python3 pipeline/music_lib.py        # score                  → assets/music
python3 pipeline/check_cues.py       # every cue exists?
pipeline/build_audio_queue.sh        # timelines + mixes      → work/chNN
python3 pipeline/build_images.py     # scene art (all cores)  → work/images
python3 pipeline/render_chapter.py   # video per chapter      → out/video
python3 pipeline/assemble.py         # the full film          → out
python3 pipeline/qa.py               # verify everything
```

### Dependencies

System: `ffmpeg`, `fluidsynth`, `fluid-soundfont-gm`, `musescore-general-soundfont`, `sox`, DejaVu fonts.
Python: `kokoro-onnx onnxruntime soundfile numpy scipy pillow opencv-python-headless pedalboard pyloudnorm pretty_midi num2words sherpa-onnx`.

Two model files are fetched once to `/opt/models`:

* Kokoro-82M TTS — `kokoro-v1.0.onnx` + `voices-v1.0.bin` (GitHub release, Apache-2.0)
* Whisper small.en ONNX — for the speech QA pass only (GitHub release)

Recorded sound sources come from Ubuntu packages (`megaglest-data`, `warzone2100-data`, `openarena-data`),
used under their own free licences; everything else is synthesised here.

## Notes on the content

The story is fiction and is written as a **tragedy for everyone except its protagonist**. Adolf is
portrayed as the monster the record shows: his crimes are named, never detailed, never justified and
never made attractive; no symbols or slogans appear. Real sitting leaders appear only as brief,
non-defamatory world texture, and after 2033 all officeholders are fictional. Every agent, officer,
scientist and executive is invented, as is every rival company. All biology in the story is fictional
and deliberately non-specific — there are no methods anywhere in these 100,000 words.
