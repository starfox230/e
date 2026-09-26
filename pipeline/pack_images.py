"""Convert the rendered scene art from PNG to high-quality JPEG.

The compositor reads png/jpg/webp, so replacing the PNGs with q92 JPEGs cuts the image
cache by about 80% and speeds up frame loading, with no visible difference after the
frames have been through Ken Burns, grading and x264.
"""
import os
import sys
import glob
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES = os.path.join(ROOT, 'work', 'images')


def convert(p):
    dst = p[:-4] + '.jpg'
    if not os.path.exists(dst):
        Image.open(p).convert('RGB').save(dst, quality=92, subsampling=1, optimize=True)
    os.remove(p)
    return dst


if __name__ == '__main__':
    pngs = sorted(glob.glob(os.path.join(IMAGES, '*.png')))
    if not pngs:
        print('nothing to convert')
        sys.exit(0)
    before = sum(os.path.getsize(p) for p in pngs)
    with ProcessPoolExecutor(max_workers=4) as ex:
        done = list(ex.map(convert, pngs, chunksize=8))
    after = sum(os.path.getsize(p) for p in done)
    print(f'{len(done)} images: {before/1e9:.2f} GB -> {after/1e9:.2f} GB')
