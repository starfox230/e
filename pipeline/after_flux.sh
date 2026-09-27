#!/bin/bash
# Wait for the Horde generators, fill whatever they could not produce with the local
# generator, then hand over to finish_images.sh (lift, swap, re-render, assemble, QA).
#   usage: pipeline/after_flux.sh <stamp-file> <generator-pid>...
cd "$(dirname "$0")/.."
STAMP="$1"; shift
log() { echo "[$(date +%H:%M:%S)] $*"; }
for p in "$@"; do
  log "waiting for generator pid $p"
  while kill -0 "$p" 2>/dev/null; do sleep 60; done
done
python3 - > work/images_flux/missing.txt <<'PY'
import json, glob, os
ids = sorted({s['id'] for p in glob.glob('work/ch*/timeline.json') for s in json.load(open(p))['shots']})
for i in ids:
    if not os.path.exists(f'work/images_flux/{i}.jpg'):
        print(i)
PY
M=$(wc -l < work/images_flux/missing.txt)
log "$M frames missing after the Horde run"
# (The film uses one photograph per shot, so only base frames count.)
# A handful missing means the Horde refused them; hundreds means the generators died (a
# container restart, say) and must be relaunched -- the slow local model is not a stand-in
# for the run itself.
if [ "$M" -gt 80 ]; then
  log "ABORT: too many frames missing for a fill; relaunch pipeline/gen_horde.py and this script"
  exit 1
fi
if [ "$M" -gt 0 ]; then
  log "filling them with the local generator"
  GEN_OUT=work/images_flux python3 pipeline/gen_images.py --names work/images_flux/missing.txt
fi
pipeline/finish_images.sh "" "$STAMP" work/images_flux
log "AFTER FLUX DONE"
