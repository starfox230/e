#!/bin/bash
# Wait for the audio lanes and the image pass, then run the rest of the build.
cd "$(dirname "$0")/.."

echo "[$(date +%H:%M:%S)] waiting for audio lanes"
while pgrep -f build_audio_split.sh > /dev/null; do sleep 30; done
MIXES=$(ls work/ch*/mix.wav 2>/dev/null | wc -l)
echo "[$(date +%H:%M:%S)] audio lanes finished: $MIXES/50 mixes"
if [ "$MIXES" -lt 50 ]; then
  echo "[$(date +%H:%M:%S)] WARNING: only $MIXES mixes; retrying the stragglers"
  pipeline/build_audio_split.sh 0 1
  MIXES=$(ls work/ch*/mix.wav 2>/dev/null | wc -l)
  echo "[$(date +%H:%M:%S)] after retry: $MIXES/50"
fi

echo "[$(date +%H:%M:%S)] waiting for image passes"
while pgrep -f build_images.py > /dev/null || pgrep -f chain_images.sh > /dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] images: $(ls work/images/*.png work/images/*.jpg 2>/dev/null | wc -l)"

echo "[$(date +%H:%M:%S)] packing art to jpeg"
python3 pipeline/pack_images.py

echo "[$(date +%H:%M:%S)] rendering video"
python3 pipeline/render_chapter.py --workers 4

echo "[$(date +%H:%M:%S)] assembling full film"
python3 pipeline/assemble.py

echo "[$(date +%H:%M:%S)] QA"
python3 pipeline/qa.py || true

echo "[$(date +%H:%M:%S)] RUN ALL DONE"
