#!/bin/bash
# Wait for the audio queue and the image pass to finish, then run the rest of the build.
cd "$(dirname "$0")/.."
echo "[$(date +%H:%M:%S)] waiting for audio queue"
while pgrep -f build_audio_queue.sh > /dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] audio queue finished: $(ls work/ch*/mix.wav 2>/dev/null | wc -l)/50 mixes"

echo "[$(date +%H:%M:%S)] waiting for image pass"
while pgrep -f build_images.py > /dev/null; do sleep 30; done

echo "[$(date +%H:%M:%S)] refreshing stale art"
python3 pipeline/refresh_images.py
python3 pipeline/build_images.py

echo "[$(date +%H:%M:%S)] packing art to jpeg"
python3 pipeline/pack_images.py

echo "[$(date +%H:%M:%S)] rendering video"
python3 pipeline/render_chapter.py --workers 4

echo "[$(date +%H:%M:%S)] assembling full film"
python3 pipeline/assemble.py

echo "[$(date +%H:%M:%S)] QA"
python3 pipeline/qa.py || true

echo "[$(date +%H:%M:%S)] RUN ALL DONE"
