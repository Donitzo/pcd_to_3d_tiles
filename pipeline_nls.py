'''
Module Name: pipeline_nls

Description:
A complete pipeline for generating 3D tiles from point cloud data (LAS/LAZ) and satellite images from the National Land Survey of Finland.

Author: Donitz
License: MIT
Repository: https://github.com/Donitzo/pcd_to_3d_tiles
'''

import argparse
import configparser
import glob
import laspy
import numpy as np
import os
import sys

from functools import reduce
from pathlib import Path
from PIL import Image, ImageEnhance
from tm35fin import MapTile
from tqdm import tqdm

from pcd_to_3d_tiles import pcd_to_3d_tiles

# Increase the maximum image size for Pillow (don't use with insecure images)
Image.MAX_IMAGE_PIXELS = 5000000000

# Load configuration
parser = argparse.ArgumentParser(description='Convert point cloud to meshes')
parser.add_argument('--config_path', default='./data/config.ini', help='Path to the configuration INI file')
parser.add_argument('--single_tile', action='store_true', help='Create a single tile no matter the extent')
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.config_path)

print('Using config file "%s"' % args.config_path)

# Get the x,y bounding box for a TM35FIN string code
def tm35fin_bound(code):
    try:
        bound = MapTile(code).bounding_box
        return (bound[0].x, bound[0].y, bound[1].x, bound[1].y)
    except:
        sys.exit('Filename is not a valid TM35FIN code: "%s"' % code)

# Get LAS file paths and bounding boxes
las_paths = glob.glob(config.get('data_sources', 'las_path_pattern'), recursive=True)
las_bounds = [tm35fin_bound(Path(path).stem) for path in las_paths]
assert las_paths, 'No LAS files found'

# Get image file paths and bounding boxes
image_paths = glob.glob(config.get('data_sources', 'image_path_pattern'), recursive=True)
image_bounds = [tm35fin_bound(Path(path).stem) for path in image_paths]

# Determine the extent of the LAS files or use the configured extent
extent_string = config.get('tiling', 'extent', fallback=None)
if not extent_string is None:
    extent = tuple(int(i) for i in extent_string.split(','))
else:
    extent = reduce(lambda acc, val: (
        min(acc[0], val[0]), min(acc[1], val[1]),
        max(acc[2], val[2]), max(acc[3], val[3])), las_bounds)

# Calculate the number of tiles needed to cover the extent
tile_size = config.getint('tiling', 'tile_size')
tile_padding = config.getint('tiling', 'tile_padding')

n_tiles_x = 1 if args.single_tile else int(np.ceil((extent[2] - extent[0]) / tile_size))
n_tiles_y = 1 if args.single_tile else int(np.ceil((extent[3] - extent[1]) / tile_size))

print('%i LAS files and %i images files found' % (len(las_paths), len(image_paths)))
print('Tile count %ix%i of size %i m' % (n_tiles_x, n_tiles_y, tile_size))
print('Full bounding box in TM35FIN coordinates [Min X=%i, Min Y=%i, Max X=%i, Max Y=%i]' % extent)
print('Creating 3D models from map tiles using point cloud data')

# Process each tile in the grid
for tx_i in range(n_tiles_x):
    for ty_i in range(n_tiles_y):
        # Calculate the bounds for the current tile
        x_min = extent[0] + tx_i * tile_size
        x_max = x_min + tile_size
        y_min = extent[1] + ty_i * tile_size
        y_max = y_min + tile_size

        tile_bound = (x_min, y_min, x_max, y_max)

        print('\nProcessing tile %ix%i of %ix%i, bounding box %s' % (
            tx_i, ty_i, n_tiles_x - 1, n_tiles_y - 1, '[Min X=%i, Min Y=%i, Max X=%i, Max Y=%i]' % tile_bound))

        pcd_list = []

        # Process each LAS file that intersects with the current tile
        progress = tqdm(list(zip(las_paths, las_bounds)))
        for path, bound in progress:
            progress.set_description('Loading LAS file "%s"' % os.path.basename(path))

            # Skip LAS files that do not intersect with the tile
            if x_min > bound[2] + tile_padding or x_max < bound[0] - tile_padding or\
                y_min > bound[3] + tile_padding or y_max < bound[1] - tile_padding:
                continue

            # Read LAS file and convert to PCD format
            las = laspy.read(path)
            pcd = np.column_stack([las.header.offset + np.column_stack([las.X, las.Y, las.Z]) * las.header.scale , las.classification])

            # Filter the points
            pcd = pcd[
                (pcd[:, 0] >= x_min - tile_padding) & (pcd[:, 0] <= x_max + tile_padding) &
                (pcd[:, 1] >= y_min - tile_padding) & (pcd[:, 1] <= y_max + tile_padding)]

            if pcd.shape[0] > 0 and np.all(pcd.ptp(0)[:2] > tile_padding * 2 + 0.1):
                pcd_list.append(pcd)

        if not pcd_list:
            print('No LAS files matching extent')
            continue

        tile_image = None

        x_density = None
        y_density = None

        # Process each image file that intersects with the current tile
        progress = tqdm(list(zip(image_paths, image_bounds)))
        for path, bound in progress:
            progress.set_description('Loading image file "%s"' % os.path.basename(path))

            # Skip image files that do not intersect with the tile
            if x_min > bound[2] or x_max < bound[0] or\
                y_min > bound[3] or y_max < bound[1]:
                continue

            # Load the image
            image = Image.open(path).convert('RGBA')

            if tile_image is None:
                # Determine the pixel dimensions of the tile_image based on the first image's pixel density
                x_density = image.size[0] / (bound[2] - bound[0])
                y_density = image.size[1] / (bound[3] - bound[1])

                tile_width = int(np.rint(x_density * tile_size))
                tile_height = int(np.rint(y_density * tile_size))

                tile_image = Image.new('RGB', (tile_width, tile_height), 0xff00ff)

            # Calculate the position to paste the current image onto the tile_image
            paste_x = int(np.rint((bound[0] - x_min) * x_density))
            paste_y = int(np.rint((y_max - bound[3]) * y_density))

            tile_image.paste(image, (paste_x, paste_y))

        # Enhance image
        if not tile_image is None:
            brightness = config.getfloat('image', 'brightness')
            contrast = config.getfloat('image', 'contrast')
            color_enhance = config.getfloat('image', 'color_enhance')

            tile_image = ImageEnhance.Brightness(tile_image).enhance(brightness)
            tile_image = ImageEnhance.Contrast(tile_image).enhance(contrast)
            tile_image = ImageEnhance.Color(tile_image).enhance(color_enhance)

        # Combine PCDs from all relevant LAS files
        pcd = np.vstack(pcd_list)

        # Get the model path prefix from the configuration template
        path_prefix = config.get('output', 'path_prefix')\
            .replace('<tile_x>', str(tx_i)).replace('<tile_y>', str(ty_i))\
            .replace('<min_x>', str(x_min)).replace('<min_y>', str(y_min))\
            .replace('<max_x>', str(x_max)).replace('<max_y>', str(y_max))

        # Create the landscape model from the point cloud and tile image
        pcd_to_3d_tiles(config, pcd, tile_bound, tile_image, path_prefix)

print('\nAll tiles created')
