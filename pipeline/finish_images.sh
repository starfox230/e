#!/bin/bash
# After scene-art generation: lift any too-dark frames, swap the new art in, re-render every
# chapter against it, then hand over to finish.sh for the freshness check, assembly and QA.
#   usage: pipeline/finish_images.sh <generator-pid> <stamp-file> [source-dir]
cd "$(dirname "$0")/.."
GEN_PID="$1"
STAMP="$2"
SRC="${3:-work/images_flux}"
log() { echo "[$(date +%H:%M:%S)] $*"; }

if [ -n "$GEN_PID" ]; then
  log "waiting for generator pid $GEN_PID"
  while kill -0 "$GEN_PID" 2>/dev/null; do sleep 30; done
fi

WANT=$(python3 -c "
import json, glob
ids = {s['id'] for p in glob.glob('work/ch*/timeline.json') for s in json.load(open(p))['shots']}
print(len(ids))")
# one photograph per shot: second takes (_v1) are spares and are not counted
HAVE=$(ls "$SRC"/*.jpg 2>/dev/null | grep -v -c '_v1\.jpg$')
log "generated $HAVE of $WANT frames"
if [ "$HAVE" -lt "$WANT" ]; then
  log "ABORT: generation incomplete; the procedural art is still in place and nothing was swapped"
  exit 1
fi

log "lifting frames under the brightness floor"
python3 pipeline/lift_dark.py "$SRC" || { log "ABORT: lift failed"; exit 1; }

# keep the procedural art beside the new set rather than deleting it: it is the fallback
log "swapping the generated art in"
rm -rf work/images_procedural
mv work/images work/images_procedural && mv "$SRC" work/images || { log "ABORT: swap failed"; exit 1; }

# A hold file stops the chain here, with the new art in place but nothing rendered: used when
# the soundtrack is about to change, since a render against the old mixes would be thrown away.
if [ -f work/HOLD_RENDER ]; then
  log "RENDER HELD (work/HOLD_RENDER exists): art is swapped in; render after the audio work"
  exit 0
fi

touch "$STAMP"
sleep 1
log "re-rendering 50 chapters"
python3 pipeline/render_chapter.py $(seq 1 50) --workers 4 --force || log "render reported an error; finish.sh will refuse a mixed cut"

pipeline/finish.sh "" "$STAMP"
log "FINISH IMAGES DONE"
