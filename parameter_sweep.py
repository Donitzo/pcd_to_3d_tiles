'''
Module Name: parameter_sweep

Description:
Manual Parameter optimization script.

Author: Donitz
License: MIT
Repository: https://github.com/Donitzo/pcd_to_3d_tiles
'''

import argparse
import configparser
import numpy as np
import os
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageEnhance, ImageFont

font = ImageFont.load_default(size=32)

parser = argparse.ArgumentParser(description='Sweep over configuration parameters')
parser.add_argument('sweep_type', type=int, help='Sweep type (0: Anisotropic diffusion, 1: Decimation RMSE, 2: Image enhancement)')
args = parser.parse_args()

if args.sweep_type == 0:
    print('Running anisotropic diffusion parameter sweep')

    # Define parameters
    k_list = [0.0, 0.5, 1.0, 1.5, 2.0]
    lambda_list = [0.01, 0.02, 0.05, 0.1, 0.2]

    # Sweep over anisotropic diffusion paramaters
    for k in k_list:
        for lambda_ in lambda_list:
            with tempfile.NamedTemporaryFile(delete=True, mode='w+', encoding='utf-8') as config_file:
                print('Creating config with sensitivity=%.2f and diffusion_coefficient=%.2f\n' % (k, lambda_))

                # Read the original configuration
                config = configparser.ConfigParser()
                config.read('./data/config.ini')

                # Update the parameters
                config.set('anisotropic_diffusion', 'sensitivity', str(k))
                config.set('anisotropic_diffusion', 'diffusion_coefficient', str(lambda_))
                config.set('output', 'path_prefix', './output/sweep_k=%.2f_lambda=%.2f' % (k, lambda_))
                config.set('screenshot', 'create_screenshot', 'yes')

                config.write(config_file)
                config_file.flush()

                # Launch the pipeline with the updated configuration
                subprocess.run([sys.executable, './pipeline_nls.py', '--config_path=%s' % config_file.name, '--single_tile'], check=True)

    collage_image = None

    # Sweep over screenshots
    for x, k in enumerate(k_list):
        for y, lambda_ in enumerate(lambda_list):
            # Load image
            png_path = './output/sweep_k=%.2f_lambda=%.2f_screenshot.png' % (k, lambda_)

            image = Image.open(png_path)

            # Create collage image
            if collage_image is None:
                collage_width = len(k_list)
                collage_height = len(lambda_list)
                collage_image = Image.new('RGB', (image.size[0] * collage_width, image.size[1] * collage_height))

            # Draw text
            text = 'Sensitivity: %.2f\nLearning rate:%.2f' % (k, lambda_)
            for i in range(4):
                ImageDraw.Draw(image).text((32 - i, 28 + i), text, font=font, fill='black')
            ImageDraw.Draw(image).text((30, 30), text, font=font, fill='red')

            # Paste the enhanced and annotated image into the collage
            collage_image.paste(image, (x * image.size[0], y * image.size[1]))

    # Save image
    png_path = './output/sweep_anisotropic_diffusion_collage.png'

    collage_image.save(png_path)

    print('Collage saved at "%s"' % png_path)

if args.sweep_type == 1:
    print('Running decimation parameter sweep')

    target_rmse_list = np.arange(0, 1 + 1e-6, 0.05)

    # Sweep over target RMSE for mesh decimation paramaters
    for target_rmse in target_rmse_list:
        with tempfile.NamedTemporaryFile(delete=True, mode='w+', encoding='utf-8') as config_file:
            print('Creating config with target_rmse=%.2f\n' % target_rmse)

            # Read the original configuration
            config = configparser.ConfigParser()
            config.read('./data/config.ini')

            # Update the parameters
            config.set('mesh_decimation', 'target_rmse', str(target_rmse))
            config.set('output', 'path_prefix', './output/sweep_rmse=%.2f' % target_rmse)
            config.set('screenshot', 'create_screenshot', 'yes')

            config.write(config_file)
            config_file.flush()

            # Launch the pipeline with the updated configuration
            subprocess.run([sys.executable, './pipeline_nls.py', '--config_path=%s' % config_file.name], check=True)

    collage_image = None

    # Sweep over screenshots
    for x, target_rmse in enumerate(target_rmse_list):
        # Load image
        png_path = './output/sweep_rmse=%.2f_screenshot.png' % target_rmse

        image = Image.open(png_path)

        # Create collage image
        if collage_image is None:
            collage_width = 7
            collage_height = int(np.ceil(len(target_rmse_list) / collage_width))
            collage_image = Image.new('RGB', (image.size[0] * collage_width, image.size[1] * collage_height))

        # Draw text
        text = 'Target RMSE: %.2f' % target_rmse
        for i in range(4):
            ImageDraw.Draw(image).text((32 - i, 28 + i), text, font=font, fill='black')
        ImageDraw.Draw(image).text((30, 30), text, font=font, fill='red')

        # Paste the enhanced and annotated image into the collage
        collage_image.paste(image, ((x % collage_width) * image.size[0], (x // collage_width) * image.size[1]))

    # Save image
    png_path = './output/sweep_rmse_collage.png'

    collage_image.save(png_path)

    print('Collage saved at "%s"' % png_path)

if args.sweep_type == 2:
    print('Running image enhancement sweep')

    # Define parameter range
    brightness_range = np.arange(0, 2 + 1e-6, 0.1)
    contrast_range = np.arange(1, 2 + 1e-6, 0.1)
    color_enhance_range = np.arange(1, 2 + 1e-6, 0.1)

    # Load sample image
    image = Image.open('./data/sweep/sample.png')

    # Calculate collage layout and prepare image
    collage_width = brightness_range.shape[0] * contrast_range.shape[0]
    collage_height = color_enhance_range.shape[0]
    collage_image = Image.new('RGB', (image.size[0] * collage_width, image.size[1] * collage_height))

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
