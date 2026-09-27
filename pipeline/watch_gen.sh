#!/bin/bash
# Keep both Horde generators alive. They die on a network hiccup or when the container
# restarts; each relaunch adopts its own in-flight requests from its ledger, so the cost of a
# restart is only the frames that were queued when it happened.
#   usage: nohup pipeline/watch_gen.sh > work/watch_gen.log 2>&1 &
cd "$(dirname "$0")/.."
alive() { pgrep -f "gen_horde.py --inflight 40 --base-only${1:+ --reverse}\$" > /dev/null; }
while true; do
  # stop once every base frame exists
  have=$(ls work/images_flux/*.jpg 2>/dev/null | grep -v -c '_v1\.jpg$')
  want=$(python3 -c "
import json, glob
print(len({s['id'] for p in glob.glob('work/ch*/timeline.json') for s in json.load(open(p))['shots']}))")
  if [ "$have" -ge "$want" ]; then
    echo "[$(date +%H:%M:%S)] $have/$want frames: done, watchdog exiting"
    exit 0
  fi
  if ! alive; then
    echo "[$(date +%H:%M:%S)] forward generator down at $have/$want; relaunching"
    nohup python3 pipeline/gen_horde.py --inflight 40 --base-only >> work/images_flux/gen.log 2>&1 &
  fi
  if ! alive rev; then
    echo "[$(date +%H:%M:%S)] reverse generator down at $have/$want; relaunching"
    nohup python3 pipeline/gen_horde.py --inflight 40 --base-only --reverse >> work/images_flux/gen_rev.log 2>&1 &
  fi
  sleep 60
done
