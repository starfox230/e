#!/bin/bash
# Build audio for every chapter script whose mix is missing or older than the script.
cd "$(dirname "$0")/.."
for s in script/ch*.txt; do
  n=$(basename "$s" .txt); n=${n#ch}
  mix="work/ch$n/mix.wav"
  if [ ! -f "$mix" ] || [ "$s" -nt "$mix" ]; then
    echo "[$(date +%H:%M:%S)] audio ch$n"
    python3 pipeline/audio.py "$s" || echo "FAILED ch$n"
  fi
done
echo "[$(date +%H:%M:%S)] queue done"
