#!/bin/bash
# Build chapter audio in two interleaved queues so both halves of the machine stay busy.
# Usage: build_audio_split.sh <lane 0|1> <lanes>
cd "$(dirname "$0")/.."
LANE=${1:-0}
LANES=${2:-2}
export OMP_NUM_THREADS=2
export ORT_INTRA_OP_NUM_THREADS=2
i=0
for s in script/ch*.txt; do
  n=$(basename "$s" .txt); n=${n#ch}
  if [ $((10#$n % LANES)) -ne "$LANE" ]; then continue; fi
  mix="work/ch$n/mix.wav"
  if [ ! -f "$mix" ] || [ "$s" -nt "$mix" ]; then
    echo "[$(date +%H:%M:%S)] lane$LANE audio ch$n"
    python3 pipeline/audio.py "$s" || echo "FAILED ch$n"
  fi
done
echo "[$(date +%H:%M:%S)] lane$LANE done"
