#!/bin/bash
cd "$(dirname "$0")/.."
while pgrep -f build_images.py > /dev/null; do sleep 20; done
echo "[$(date +%H:%M:%S)] base pass done; rendering alternate compositions"
python3 pipeline/build_images.py
echo "[$(date +%H:%M:%S)] variants done: $(ls work/images/*.png work/images/*.jpg 2>/dev/null|wc -l) images"
