#!/bin/bash
# Build the whole film from the scripts: audio in two lanes, then a retry pass for anything
# that failed or went stale mid-build, then video, then the full assembly and QA.
#
# Safe to re-run: every stage skips work that is already current, and the audio lanes rebuild
# any chapter whose script is newer than its mix, which is what catches a script edited while
# an earlier pass was running.
cd "$(dirname "$0")/.."
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

log "audio, two lanes"
bash pipeline/build_audio_split.sh 0 2 &
L0=$!
bash pipeline/build_audio_split.sh 1 2 &
L1=$!
wait $L0 $L1

for pass in 1 2; do
    MIX=$(ls work/ch*/mix.wav 2>/dev/null | wc -l)
    STALE=0
    for s in script/ch*.txt; do
        n=$(basename "$s" .txt); n=${n#ch}
        [ "$s" -nt "work/ch$n/mix.wav" ] && STALE=$((STALE + 1))
    done
    log "after audio pass $pass: $MIX/50 mixes, $STALE stale"
    [ "$MIX" -ge 50 ] && [ "$STALE" -eq 0 ] && break
    log "retry pass"
    bash pipeline/build_audio_split.sh 0 2 &
    bash pipeline/build_audio_split.sh 1 2 &
    wait
done

FRAMES=$(ls work/images_anime/*.jpg 2>/dev/null | wc -l)
SHOTS=$(python3 pipeline/gen_anime.py --report --upto 50 2>&1 | tail -1 | awk '{print $1}')
log "frames: $FRAMES of $SHOTS"

log "render"
python3 pipeline/render_chapter.py --workers 4 || log "render reported an error"

log "assemble"
python3 pipeline/assemble.py

log "QA"
python3 pipeline/qa.py || true

log "BUILD DONE"
