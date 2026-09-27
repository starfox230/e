#!/bin/sh
# Keep the frame generator alive. The container is recycled every so often and takes
# background work with it; this restarts the generator whenever it is not running.
# The check matches only a real python process: a shell whose command line merely
# mentions the script (the one that launched this watchdog, for instance) is not it.
cd /home/user/e
LOG=/tmp/claude-0/-home-user-e/b62efde1-8f1a-59c0-a070-04b692eaca77/scratchpad/gen_anime.log
while true; do
    if ! ps -eo comm,args | grep -E '^python3 +python3 +pipeline/gen_anime\.py' >/dev/null; then
        echo "--- $(date -u +%H:%M:%S) starting generator" >> "$LOG"
        nohup python3 pipeline/gen_anime.py --upto "${UPTO:-30}" --inflight 40 >> "$LOG" 2>&1 &
    fi
    sleep 60
done
