#!/bin/bash
# Free each chapter's mix as soon as its video is rendered and checked.
# The render needs room for fifty chapter files and then the assembled film on top, and the
# mixes are seven and a half gigabytes that nothing needs once the chapter is muxed.
cd /home/user/e
while true; do
    for f in out/video/ch*.mp4; do
        [ -f "$f" ] || continue
        n=$(basename "$f" .mp4); n=${n#ch}
        m="work/ch$n/mix.wav"
        [ -f "$m" ] || continue
        # only when the video is complete: it must be older than a minute and not growing
        if [ -z "$(find "$f" -newermt '-60 seconds')" ]; then
            python3 pipeline/qa.py "$((10#$n))" >> /tmp/claude-0/-home-user-e/b62efde1-8f1a-59c0-a070-04b692eaca77/scratchpad/qa.log 2>&1 \
                || echo "QA flagged ch$n" >> /tmp/claude-0/-home-user-e/b62efde1-8f1a-59c0-a070-04b692eaca77/scratchpad/qa.log
            rm -f "$m"
            echo "reaped ch$n (free $(df -h / | awk 'NR==2{print $4}'))"
        fi
    done
    [ "$(ls out/video/ch*.mp4 2>/dev/null | wc -l)" -ge 50 ] && [ -z "$(ls work/ch*/mix.wav 2>/dev/null)" ] && break
    sleep 60
done
echo "reaper done"
