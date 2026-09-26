#!/bin/bash
# Render every chapter, assemble the full film, then QA. Reports disk as it goes.
cd "$(dirname "$0")/.."
echo "[$(date +%H:%M:%S)] rendering 50 chapters"
python3 pipeline/render_chapter.py --workers 4
echo "[$(date +%H:%M:%S)] chapters rendered: $(ls out/video/*.mp4 2>/dev/null | wc -l)/50, $(df -h / | awk 'NR==2{print $4}') free"
echo "[$(date +%H:%M:%S)] assembling full film"
python3 pipeline/assemble.py
echo "[$(date +%H:%M:%S)] QA"
python3 pipeline/qa.py || true
echo "[$(date +%H:%M:%S)] VIDEO DONE"
