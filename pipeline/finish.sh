#!/bin/bash
# Final build: stale art refresh -> remaining art -> pack -> video -> assemble -> QA
set -e
cd "$(dirname "$0")/.."
echo "[$(date +%H:%M:%S)] refreshing stale art"
python3 pipeline/refresh_images.py
python3 pipeline/build_images.py
echo "[$(date +%H:%M:%S)] packing art"
python3 pipeline/pack_images.py
echo "[$(date +%H:%M:%S)] rendering video"
python3 pipeline/render_chapter.py --workers 4
echo "[$(date +%H:%M:%S)] assembling"
python3 pipeline/assemble.py
echo "[$(date +%H:%M:%S)] QA"
python3 pipeline/qa.py || true
echo "[$(date +%H:%M:%S)] FINISH DONE"
