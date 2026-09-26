#!/bin/bash
# Wait for the chapter render to finish, then rebuild the full film and QA it.
#   usage: pipeline/finish.sh <render-pid> <stamp-file>
# Takes the render's PID so it never matches its own command line the way a pgrep would.
cd "$(dirname "$0")/.."
RENDER_PID="$1"
STAMP="$2"

if [ -n "$RENDER_PID" ]; then
  echo "[$(date +%H:%M:%S)] waiting for render pid $RENDER_PID"
  while kill -0 "$RENDER_PID" 2>/dev/null; do sleep 20; done
fi

# Counting the files is not enough: out/video keeps the previous cut's chapters, so a render
# that dies on chapter 2 still leaves 50 of them sitting there and the count sails through.
# Each chapter has to be newer than the stamp laid down when this render started.
if [ -z "$STAMP" ] || [ ! -f "$STAMP" ]; then
  echo "[$(date +%H:%M:%S)] ABORT: no stamp file, cannot tell fresh renders from stale ones"
  exit 1
fi
stale=0
missing=0
for i in $(seq -w 1 50); do
  f="out/video/ch$i.mp4"
  if [ ! -f "$f" ]; then missing=$((missing+1))
  elif [ "$f" -ot "$STAMP" ]; then stale=$((stale+1)); fi
done
echo "[$(date +%H:%M:%S)] chapters: $((50-missing-stale)) fresh, $stale stale, $missing missing"
if [ "$stale" -ne 0 ] || [ "$missing" -ne 0 ]; then
  echo "[$(date +%H:%M:%S)] ABORT: not assembling a mixed cut"
  exit 1
fi

# the previous cut is derived from these chapters and rebuilds in minutes, so it goes now
# rather than competing for the last few gigabytes of the disk allowance
rm -f out/umbrella_full.mp4
echo "[$(date +%H:%M:%S)] free after clearing the old cut: $(df -h / | awk 'NR==2{print $4}')"

echo "[$(date +%H:%M:%S)] assembling"
python3 pipeline/assemble.py || { echo "[$(date +%H:%M:%S)] ASSEMBLE FAILED"; exit 1; }

echo "[$(date +%H:%M:%S)] QA"
python3 pipeline/qa.py
echo "[$(date +%H:%M:%S)] FINISH DONE"
