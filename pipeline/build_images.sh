#!/bin/bash
cd "$(dirname "$0")/.."
for n in $(seq -w 1 50); do
  echo "[$(date +%H:%M:%S)] images ch$n"
  python3 pipeline/art.py --chapter $((10#$n)) || echo "FAILED ch$n"
done
echo "[$(date +%H:%M:%S)] images done"
