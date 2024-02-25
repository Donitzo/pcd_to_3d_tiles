'''
Module Name: parameter_sweep

Description:
Manual hyper-parameter optimization script.

Sweep type 0: Iterate anistropic diffusion parameters in "./data/config.json"
Sweep type 1: Generate a collage of image enhancement parameters

Author: Donitz
License: MIT
'''

import argparse
import configparser
import numpy as np
import os
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageEnhance, ImageFont

parser = argparse.ArgumentParser(description='Sweep over configuration parameters')
parser.add_argument('sweep_type', type=int, help='Sweep type (0: Anistropic diffusion, 1: Image enhancement)')
args = parser.parse_args()

if args.sweep_type == 0:
    with tempfile.NamedTemporaryFile(delete=True, mode='w+', encoding='utf-8') as config_file:
        print('Running anistropic diffusion parameter sweep')

        # Read the original configuration
        config = configparser.ConfigParser()
        config.read('./data/config.ini')

        # Sweep over anistropic diffusion paramaters
        for sensitivity in [1.0, 2.0, 3.0, 4.0]:
            for diffusion_coefficient in [0.01, 0.02, 0.05, 0.1, 0.2]:
                print('Creating config with sensitivity %.2f and diffusion_coefficient %.2f\n' % (sensitivity, diffusion_coefficient))

                # Update the parameter in the config
                config.set('anisotropic_diffusion', 'sensitivity', str(sensitivity))
                config.set('anisotropic_diffusion', 'diffusion_coefficient', str(diffusion_coefficient))

                config.set('output', 'model_path_prefix', './output/sweep_k=%.2f_lambda=%.2f' % (sensitivity, diffusion_coefficient))

                config.write(config_file)
                config_file.flush()

                # Launch the pipeline with the updated configuration
                subprocess.run([sys.executable, './pipeline_nls.py', '--config_path=%s' % config_file.name], check=True)

if args.sweep_type == 1:
    print('Running image enhancement sweep')

    # Define parameter range
    brightness_range = np.arange(0, 2 + 1e-6, 0.1)
    contrast_range = np.arange(1, 2 + 1e-6, 0.1)
    color_enhance_range = np.arange(1, 2 + 1e-6, 0.1)

    # Load sample image
    image = Image.open('./data/sweep/sample_small.png')

    # Calculate collage layout and prepare image
    collage_width = brightness_range.shape[0] * contrast_range.shape[0]
    collage_height = color_enhance_range.shape[0]
    collage_image = Image.new('RGB', (image.size[0] * collage_width, image.size[1] * collage_height))

    # Prepare to draw text
    font = ImageFont.load_default(size=32)

    # Sweep over image enhancement paramaters
    x, y = 0, 0
    for color_enhance in color_enhance_range:
        for brightness in brightness_range:
            for contrast in contrast_range:
                print('Creating sample image with brightness %.1f and contrast %.1f and color enhance %.1f' % (brightness, contrast, color_enhance))

                # Enhance image
                enhanced_image = ImageEnhance.Brightness(image).enhance(brightness)
                enhanced_image = ImageEnhance.Contrast(enhanced_image).enhance(contrast)
                enhanced_image = ImageEnhance.Color(enhanced_image).enhance(color_enhance)

                # Draw text
                text = 'B:%.1f\nC:%.1f\nCE:%.1f' % (brightness, contrast, color_enhance)
                for i in range(4):
                    ImageDraw.Draw(enhanced_image).text((32 - i, 28 + i), text, font=font, fill='black')
                ImageDraw.Draw(enhanced_image).text((30, 30), text, font=font, fill='red')

                # Paste the enhanced and annotated image into the collage
                collage_image.paste(enhanced_image, (x * image.size[0], y * image.size[1]))

                # Update position for the next image
                x += 1
                if x >= collage_width:
                    x = 0
                    y += 1

    # Save image
    png_path = './output/sample_collage.png'

    collage_image.save(png_path)

    print('Collage saved at "%s"' % png_path)
