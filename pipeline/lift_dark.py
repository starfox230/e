"""Lift generated frames that came out too dark to read on screen.

The generated art is photographic, and a night scene can land where the brightest 1% of the
frame is still near black -- a shot holds for the better part of a minute, so that is a
minute of nothing to look at. The fix is the same gamma lift art.grade() applies to the
procedural frames: both endpoints stay put, the tones between come up. Only frames under the
floor are touched, so the model's own exposure stands everywhere else.

Usage: python3 pipeline/lift_dark.py [image-dir]      (default work/images)
"""
import os
import sys
import glob
import math

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from art import FLOOR_P99, LIFT_TO_P99          # one definition of "too dark" for the film

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def luma_p99(a):
    return float(np.percentile(a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114, 99))


def main(d):
    lifted = 0
    files = sorted(glob.glob(os.path.join(d, '*.jpg')))
    for f in files:
        a = np.asarray(Image.open(f).convert('RGB'), np.float32) / 255.0
        bright = luma_p99(a)
        if not 1e-3 < bright < FLOOR_P99:
            continue
        gamma = max(0.45, math.log(LIFT_TO_P99) / math.log(bright))
        out = np.clip(a, 0, 1) ** gamma
        tmp = os.path.join(d, '.' + os.path.basename(f))
        Image.fromarray((out * 255 + 0.5).astype(np.uint8)).save(tmp, quality=92, subsampling=1, optimize=True)
        os.replace(tmp, f)
        lifted += 1
        print(f'  lifted {os.path.basename(f)}: p99 {bright * 255:.0f} -> {luma_p99(out) * 255:.0f}')
    print(f'{lifted} of {len(files)} frames lifted above the floor ({FLOOR_P99 * 255:.0f}/255)')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'work', 'images'))
