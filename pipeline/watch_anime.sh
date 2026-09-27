#!/bin/sh
# Keep the frame generators alive. The container is recycled every so often and takes
# background work with it; this restarts whichever generator is not running.
# Two processes on two checkpoints: the Horde's anonymous queue for any one model drains
# at about two frames a minute, so the second one roughly doubles the rate. They work from
# opposite ends of the shot list and skip frames already on disk, so they never collide.
cd /home/user/e
LOG=/tmp/claude-0/-home-user-e/b62efde1-8f1a-59c0-a070-04b692eaca77/scratchpad/gen_anime.log
UPTO=${UPTO:-33}
while true; do
    if ! ps -eo comm,args | grep -E '^python3 +python3 +pipeline/gen_anime\.py --upto [0-9]+ --inflight [0-9]+$' >/dev/null; then
        echo "--- $(date -u +%H:%M:%S) starting forward generator" >> "$LOG"
        ANIME_MODEL=aam nohup python3 pipeline/gen_anime.py --upto "$UPTO" --inflight 60 >> "$LOG" 2>&1 &
    fi
    if ! ps -eo comm,args | grep -E '^python3 +python3 +pipeline/gen_anime\.py .*--reverse' >/dev/null; then
        echo "--- $(date -u +%H:%M:%S) starting reverse generator" >> "$LOG"
        ANIME_MODEL=animagine nohup python3 pipeline/gen_anime.py --upto "$UPTO" --inflight 60 --reverse >> "$LOG" 2>&1 &
    fi
    sleep 60
done
