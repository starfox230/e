#!/bin/bash
cd "$(dirname "$0")/.."
prev_a=0; imgdone=0
while true; do
  a=$(ls work/ch*/mix.wav 2>/dev/null | wc -l)
  i=$(ls work/images/*.png 2>/dev/null | wc -l)
  if [ "$a" -ge $((prev_a+5)) ]; then echo "audio: $a/50 chapters mixed"; prev_a=$a; fi
  if [ "$imgdone" = "0" ] && grep -q "images done" images.log 2>/dev/null; then echo "images: all $i rendered"; imgdone=1; fi
  if grep -qE "Traceback|FAILED" audio_queue3.log 2>/dev/null; then echo "AUDIO ERROR"; grep -hE "FAILED|Error" audio_queue3.log | tail -2; fi
  if grep -qE "Traceback" images.log 2>/dev/null; then echo "IMAGE ERROR"; grep -A3 Traceback images.log | tail -4; fi
  if [ "$a" -ge 50 ] && [ "$imgdone" = "1" ]; then echo "ALL BUILDS COMPLETE"; break; fi
  sleep 60
done
