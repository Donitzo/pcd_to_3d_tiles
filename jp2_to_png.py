'''
Module Name: jp2_to_png

Description:
Converts all the JPEG2000 images in the "./data/orto/" directory into PNG images, which are faster to read.

Author: Donitz
License: MIT
Repository: https://github.com/Donitzo/pcd_to_3d_tiles
'''

import glob
import os

from PIL import Image

# Increase the maximum image size for Pillow (don't use with insecure images)
Image.MAX_IMAGE_PIXELS = 5000000000

jp2_paths = glob.glob('./data/orto/**/*.jp2', recursive=True)

# Convert every JPEG2000 file in "./data/orto" into PNG images
for path in jp2_paths:
    png_path = path.rsplit('.', 1)[0] + '.png'
    if not os.path.exists(png_path):
        print('Converting "%s" to PNG' % path)

        with Image.open(path) as image:
            image.save(png_path, 'PNG')
