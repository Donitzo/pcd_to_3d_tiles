'''
Module Name: pcd_to_3d_tiles

Description:
Handles the conversion from point cloud data to 3D meshes (tiles).

This involves multiple processing steps configured in the provided configuration file.

Steps:
  - Outlier_removal using nearest neighbor counting
  - Removal of ground points from vegetation using Delaunay triangulation
  - Smoothing vegetation height using nearest neighbor aggregation and percentile height
  - Anistropic diffusion smoothing of point clouds
  - Model triangulation
  - Decimation using binary search

Example: pcd_to_3d_tiles(config, pcd, bound, image, path_prefix)

Author: Donitz
License: MIT
'''

import json
import numpy as np
import open3d as o3d
import os
import plyfile
import sys
import trimesh

from pathlib import Path
from PIL import Image
from scipy.spatial import cKDTree, Delaunay
from tqdm import tqdm

class UnsupportedVersion(Exception):
    pass

MIN_VERSION, VERSION_LESS_THAN = (3, 9), (4, 0)
if sys.version_info < MIN_VERSION or sys.version_info >= VERSION_LESS_THAN:
    raise UnsupportedVersion('requires Python %s,<%s' % ('.'.join(map(str, MIN_VERSION)), '.'.join(map(str, VERSION_LESS_THAN))))

VEGETATION_CLASSES = [3, 4, 5]

def _outlier_removal(config, pcd):
    '''
        Remove outliers based on nearest neighbor counting within the configured radius.
    '''

    n_neighbors = config.getint('outlier_detection', 'min_neighbors')
    distance = config.getfloat('outlier_detection', 'max_neighbor_distance')

    distances = cKDTree(pcd[:, :3], leafsize=32).query(pcd[:, :3], k=n_neighbors + 1, distance_upper_bound=distance)[0]
    is_outlier = np.sum(distances != np.inf, axis=1) - 1 < n_neighbors

    print('Removed %i outliers (%.4f%% of point cloud)' % (is_outlier.sum(), 100.0 * is_outlier.sum() / pcd.shape[0]))

    return pcd[~is_outlier]

def _is_point_in_triangle_2d(points, triangle):
    '''
        Get a boolean vector whether each point is inside the triangle (disregarding z).
    '''

    a, b, c = triangle

    def sign(p1, p2):
        return (points[:, 0] - p2[0]) * (p1[1] - p2[1]) - (p1[0] - p2[0]) * (points[:, 1] - p2[1])

    d1 = sign(a, b)
    d2 = sign(b, c)
    d3 = sign(c, a)

    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)

    return ~(has_neg & has_pos)

def _remove_ground_points_from_vegetation(config, pcd):
    '''
        Remove ground points from vegetation points using triangulation.
    '''

    max_edge_length = config.getfloat('vegetation', 'max_vegetation_filter_face_edge_length')

    print('Removing ground points located inside dense vegetation')

    # Get points with the vegetation and ground classes
    is_vegetation = np.isin(pcd[:, 3], VEGETATION_CLASSES)
    pcd_v = pcd[is_vegetation]

    if pcd_v.shape[0] < 4:
        print('->No vegetation found')

        return pcd

    ground_indices = np.where(~is_vegetation)[0]
    pcd_g = pcd[ground_indices]

    # Triangulate vegetation faces from top-down using Delaunay triangulation
    faces = Delaunay(pcd_v[:, :2]).simplices

    # Filter faces based on max edge length
    vertices = pcd_v[faces, :2]
    side_lengths = np.sqrt(np.sum(np.square(vertices - np.roll(vertices, -1, axis=1)), axis=2))

    faces = faces[np.all(side_lengths <= max_edge_length, axis=1)]

    # Get triangles and triangle points from vegetation faces (every third point forms a triangle)
    triangles = pcd_v[faces, :2]
    triangle_points = pcd_v[faces.flatten(), :2]

    # Get the ground points near the triangle points
    point_indices = cKDTree(pcd_g[:, :2], leafsize=32).query_ball_point(triangle_points, r=max_edge_length)

    # Merge every third point index list into a triangle index list
    triangle_indices = [list(set().union(*point_indices[i:i + 3])) for i in range(0, len(point_indices), 3)]

    # Check if ground points are located inside vegetation faces
    in_vegetation_mesh = np.zeros(pcd_g.shape[0], dtype=np.bool_)
    for triangle, indices in zip(triangles, triangle_indices):
        in_vegetation_mesh[indices] |= _is_point_in_triangle_2d(pcd_g[indices, :2], triangle)

    # Exclude ground points inside vegetation
    exclude = np.zeros(pcd.shape[0], dtype=np.bool_)
    exclude[ground_indices[in_vegetation_mesh]] = True

    print('->Removed %i ground points (%.4f%% of point cloud)' % (exclude.sum(), 100.0 * exclude.sum() / pcd.shape[0]))

    return pcd[~exclude]

def _smooth_vegetation_height(config, pcd):
    '''
        Aggregate/smooth the height of vegetation based on neighboring vegetation points.
    '''

    distance = config.getfloat('vegetation', 'aggregate_distance')
    aggregate_percentile = config.getfloat('vegetation', 'aggregate_percentile')

    print('Aggregating/smoothing vegetation height')

    # Get points with a vegetation class
    vegetation_indices = np.where(np.isin(pcd[:, 3], VEGETATION_CLASSES))[0]
    pcd_v = pcd[vegetation_indices]

    if pcd_v.shape[0] < 4:
        print('->No vegetation found')

        return pcd

    # Get vegetation neighbors for all the vegetation points
    indices = cKDTree(pcd_v[:, :2], leafsize=32).query_ball_point(pcd_v[:, :2], r=distance)

    # Create a new point cloud with the aggregate vegetation height
    pcd_new = pcd.copy()

    for i, point_indices in enumerate(indices):
        # Skip if no neighbors are found
        if not point_indices:
            continue

        # Calculate the specified percentile for the heights of the neighborhood vegetation
        z = np.percentile(pcd_v[point_indices, 2], aggregate_percentile)

        # Update the height in the new point cloud for the current vegetation point
        pcd_new[vegetation_indices[i], 2] = z

    aad = np.mean(np.abs(pcd_new[vegetation_indices, 2] - pcd[vegetation_indices, 2]))
    print('->Vegetation points adjusted an average absolute deviation of %.2f m in height' % aad)

    return pcd_new

def _anistropic_diffusion(config, pcd):
    '''
        Smooth point cloud using anistropic diffusion.
    '''

    iterations = config.getint('anisotropic_diffusion', 'iterations')
    n_neighbors = min(config.getint('anisotropic_diffusion', 'neighbors'), pcd.shape[0] - 1)
    k = config.getfloat('anisotropic_diffusion', 'sensitivity')
    lambda_ = config.getfloat('anisotropic_diffusion', 'diffusion_coefficient')

    pcd_new = pcd.copy()

    progress = tqdm(range(iterations))
    progress.set_description('Applying anistropic diffusion smoothing')

    for _ in progress:
        # Query the nearest neighbors for each point, excluding the point itself
        neighbors = cKDTree(pcd_new[:, :3], leafsize=32).query(pcd_new[:, :3], k=n_neighbors + 1)[1][:, 1:]

        # Calculate the differences between points and their neighbors
        diffs = pcd_new[:, :3][neighbors].astype(np.float32) - pcd_new[:, :3][:, np.newaxis, :].astype(np.float32)
        del neighbors

        # Compute the gradient magnitudes and the edge-stopping function
        gradient_magnitudes = np.linalg.norm(diffs, axis=2)
        c = np.exp(-(gradient_magnitudes / k) ** 2)
        del gradient_magnitudes

        # Compute the diffusion updates as a weighted sum of the differences and pdate the point cloud
        pcd_new[:, :3] += lambda_ * np.sum(c[:, :, np.newaxis] * diffs, axis=1)
        del diffs
        del c

    aad = np.mean(np.abs(pcd_new[:, 2] - pcd[:, 2]))
    print('->Points adjusted an average absolute deviation of %.2f m' % aad)

    return pcd_new

def _create_model(config, pcd, bound, image, obj_path):
    '''
        Create model mesh from a point cloud and an image.
        The mesh is triangulated top-down using Delaunay triangulation.
        The mesh is decimated until the error between the mesh and point cloud is close to the target Root-mean-square error.
    '''

    x_min, y_min, x_max, y_max = bound
    tile_size_x = x_max - x_min
    tile_size_y = y_max - y_min

    # Get the .5 percentile height from the landscape
    min_height = np.percentile(pcd[:, 2], 0.5)

    # Pad the data with exterior corner points at the min height to prevent missing gaps due to missing points.
    # A large distance is used to reduce sloping.
    corners = np.array([
        [-100000, -100000, min_height, 1],
        [-100000, 100000 + tile_size_y, min_height, 1],
        [100000 + tile_size_x, 100000 + tile_size_y, min_height, 1],
        [100000 + tile_size_x, -100000, min_height, 1],
    ])

    pcd = np.vstack([pcd, corners])

    # Triangulate faces from top-down using Delaunay triangulation
    faces = Delaunay(pcd[:, :2]).simplices

    # Change coordinate system for vertices
    vertices = pcd[:, [0, 2, 1]]
    vertices[:, 2] = -vertices[:, 2]

    # Find the optimal decimation factor for the mesh using binary search
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    new_mesh = mesh

    target_rmse = config.getfloat('mesh_decimation', 'target_rmse', fallback=None)
    search_iterations = config.getint('mesh_decimation', 'binary_search_iterations')

    decimation_factor = 0.5
    decimation_factor_step = 0.25

    progress = tqdm(range(search_iterations))
    progress.set_description('Finding mesh decimation factor')

    for _ in progress:
        # Decimate mesh
        target_faces = int(mesh.faces.shape[0] * decimation_factor)
        new_mesh = mesh.simplify_quadric_decimation(target_faces)

        # Find the closest distance between the decimated mesh and the point cloud
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(new_mesh.as_open3d))
        closest = scene.compute_closest_points(o3d.core.Tensor.from_numpy(vertices.astype(np.float32)))['points'].numpy()

        # Calculate RMSE
        distances = np.sqrt(np.sum((vertices - closest) ** 2, axis=1))
        rmse = np.sqrt(np.mean(distances ** 2))

        progress.set_description('Mesh decimated with a factor of %.4f and RMSE %.2f m (Target RMSE %.2f m)' % (decimation_factor, rmse, target_rmse))

        # Update decimation facotr
        decimation_factor += decimation_factor_step * (1 if rmse > target_rmse else -1)
        decimation_factor_step /= 2

    print('->Decimated mesh has %i faces (%.2f%% reduction)' % (new_mesh.faces.shape[0], 100 - float(new_mesh.faces.shape[0]) / mesh.faces.shape[0] * 100))

    # Slice edges off mesh
    new_mesh = new_mesh.slice_plane([0, 0, 0], [1, 0, 0])
    new_mesh = new_mesh.slice_plane([tile_size_x, 0, 0], [-1, 0, 0])
    new_mesh = new_mesh.slice_plane([0, 0, 0], [0, 0, -1])
    new_mesh = new_mesh.slice_plane([0, 0, -tile_size_y], [0, 0, 1])

    print('->Cropped mesh has %i faces' % new_mesh.faces.shape[0])

    # Create UV coordinates
    uv = new_mesh.vertices[:, [0, 2]].copy()
    uv[:, 0] = uv[:, 0] / tile_size_x
    uv[:, 1] = -uv[:, 1] / tile_size_y

    # Add material
    if not image is None:
        new_mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=image)
        new_mesh.visual.material.name = Path(obj_path).stem
        new_mesh.visual.material.specular = [0.0, 0.0, 0.0]
        new_mesh.visual.material.diffuse = [0.0, 1.0, 0.0]

    # Export mesh
    new_mesh.export(obj_path, mtl_name='%s.mtl' % Path(obj_path).stem)

    print('Model saved at "%s"' % obj_path)

def _take_screenshot(config, obj_path, png_path):
    '''
        Take a screenshot of the mesh. Uses the camera parameters in "camera_parameters_json_path" if the file exists.
    '''

    json_path = config.get('screenshot', 'camera_parameters_json_path', fallback=None)

    if not json_path is None:
        # Load previously saved camera parameters from JSON (can be created using the Open3D visualizer)
        with open(json_path, 'r') as f:
            view = json.load(f)
            intrinsic = view['intrinsic']
            camera_parameters = o3d.camera.PinholeCameraParameters()
            camera_parameters.extrinsic = np.array(view['extrinsic']).reshape(4, 4, order='F')
            camera_parameters.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                intrinsic['width'], intrinsic['height'],
                np.array(intrinsic['intrinsic_matrix']).reshape(3, 3, order='F'))
            width = intrinsic['width']
            height = intrinsic['height']
    else:
        camera_parameters = None
        width = 1920
        height = 1080

    # Initialize the Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)

    # Load the textured mesh again
    mesh_o3d = o3d.io.read_triangle_mesh(obj_path, True)

    # Add the textured mesh to the visualizer
    vis.add_geometry(mesh_o3d)

    # Configure camera
    if not camera_parameters is None:
        vis.get_view_control().convert_from_pinhole_camera_parameters(camera_parameters, True)

    # Render the scene to capture the textured mesh
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(png_path, do_render=True)
    vis.destroy_window()

    print('Screenshot saved')

def _save_point_cloud(config, pcd, ply_path):
    '''
        Save the point cloud as a PLY file.
    '''

    # Create vertex attributes
    vertex = np.empty(pcd.shape[0], dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ])

    # Assign coordinates
    vertex['x'] = pcd[:, 0].astype(np.float32)
    vertex['y'] = pcd[:, 1].astype(np.float32)
    vertex['z'] = pcd[:, 2].astype(np.float32)

    # Assign colors from class
    class_colors = np.array([[0, 0, 0], [255, 0, 0], [255, 255, 0], [0, 128, 0], [0, 192, 0], [0, 255, 0], [255, 255, 0]], dtype=np.ubyte)
    colors = class_colors[np.minimum(pcd[:, 3].astype(np.short), class_colors.shape[0] - 1)]

    vertex['red'] = colors[:, 0]
    vertex['green'] = colors[:, 1]
    vertex['blue'] = colors[:, 2]

    # Write PLY file
    ply = plyfile.PlyData([plyfile.PlyElement.describe(vertex, 'vertex')], text=True)
    ply.write(ply_path)

    print('Point cloud model saved at "%s"' % ply_path)

def pcd_to_3d_tiles(config, pcd, bound, image, path_prefix):
    '''
    This function takes a point cloud as input and processes it to produce a 3D landscape tile.

    Parameters:
    - config (ConfigParser): Configuration file loaded using load_config.
    - pcd (np.array): The input point cloud to be processed in the shape of (N, 4) where the fourth dimension is LAS class.
    - bound (tuple or list): Specifies the bounding box for the landscape model in the format (min x, min y, max x, max y).
      Points outside the bounding box will still influence the model's shape.
    - image (PIL.Image): The image covering the point cloud bound.
    - path_prefix (str): The path prefix after template replacement.
    '''

    # Create the output path if necessary
    directory = os.path.dirname(path_prefix) or '.'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Offset point cloud towards origin to reduce floating-point inaccuracy
    pcd[:, 0] -= bound[0]
    pcd[:, 1] -= bound[1]

    # Save the original point cloud for further analysis
    if config.getboolean('output', 'save_point_cloud'):
        ply_path = '%s_pcd_original.ply' % path_prefix

        _save_point_cloud(config, pcd, ply_path)

    # Remove outliers
    pcd = _outlier_removal(config, pcd)

    # There must be at least 4 points remaining to generate meshes
    assert pcd.shape[0] >= 4, 'Not enough points remaining'

    # Remove ground points located in vegetation
    pcd = _remove_ground_points_from_vegetation(config, pcd)

    # Smooth vegetation height
    pcd = _smooth_vegetation_height(config, pcd)

    # Smooth the point cloud using anistropic diffusion
    pcd = _anistropic_diffusion(config, pcd)

    # There may not be any non-finite values in the point cloud
    assert not np.isnan(pcd).any() and not np.isinf(pcd).any(), 'Point cloud data contains NaN or infinite values'

    # Save the processed point cloud
    if config.getboolean('output', 'save_point_cloud'):
        ply_path = '%s_pcd_processed.ply' % path_prefix

        _save_point_cloud(config, pcd, ply_path)

    obj_path = '%s_mesh.obj' % path_prefix

    # Create the mesh
    _create_model(config, pcd, bound, image, obj_path)

    # Create a screenshot of the mesh
    if config.getboolean('screenshot', 'create_screenshot'):
        png_path = '%s_screenshot.png' % path_prefix

        _take_screenshot(config, obj_path, png_path)
