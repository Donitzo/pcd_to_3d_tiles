'''
Module Name: visualize mesh

Description:
Helper script to visualize a mesh using Open3D.
Useful to output camera parameters for screenshots.

Author: Donitz
License: MIT
'''

import argparse
import open3d as o3d

parser = argparse.ArgumentParser(description='Visualize OBJ Mesh')
parser.add_argument('obj_path', help='Path to the OBJ file')
args = parser.parse_args()

# Read mesh
mesh = o3d.io.read_triangle_mesh(args.obj_path, True)

# Visualize mesh
o3d.visualization.draw_geometries([mesh])

print('\nYou can press "p" to capture camera parameters from the current view.')
