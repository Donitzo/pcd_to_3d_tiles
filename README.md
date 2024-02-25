# PCD to 3D tiles

This is a simple Python pipeline for transforming point cloud data, specifically in the [LAS format](https://en.wikipedia.org/wiki/LAS_file_format), into textured 3D landscape tiles. Designed with simplicity in mind, the pipeline focuses on generating landscape tiles that are very lightweight.

## Disclaimer

The landscape tiles generated by this pipeline serve as a basic visual representation of the terrain, prioritizing aesthetic appeal over geographical accuracy. As such, the output should not be used for measurements or analysis where exact dimensions and positions are important.

## Installation

Begin by cloning the GitHub repository to your local machine:

`git clone https://github.com/Donitzo/pcd_to_3d_tiles`

and install the requirements in your Python environment of choice:

`pip install -r requirements.txt`

Reasoning behind the requirements:

  - `laspy[lazrs]`: Enables reading LAS and LAZ point cloud files.
  - `open3d`: A 3D library offering comprehensive mesh and point cloud operations for visualization and processing.
  - `pillow`: For reading/writing images.
  - `plyfile`: For writing PLY files (used for saving point clouds).
  - `scipy`: For using spatial models such as Delaunay and cKDTree.
  - `shapely`: Used by trimesh.
  - `tm35fin`: A simple module which can be used to and from TM35FIN.
  - `trimesh`: Used for reading and writing model files with texture support.
  - `tqdm`: For progress bars.

## Point cloud format

The point cloud should be pre-classified according to the classification standards defined by the [American Society for Photogrammetry and Remote Sensing (ASPRS)](https://www.asprs.org/wp-content/uploads/2019/03/LAS_1_4_r14.pdf).

The following table outlines some common classes:

| Class | Name             |
|-------|------------------|
| 0     | Never classified |
| 1     | Unassigned       |
| 2     | Ground           |
| 3     | Low Vegetation   |
| 4     | Medium Vegetation|
| 5     | High Vegetation  |
| 6...  | Various other    |

This pipeline specifically processes points classified as Low, Medium, or High Vegetation in certain preprocessing steps. All other classes are treated as ground.

## Configuration

The pipeline's behavior is controlled by a configuration file, by default `data/config.json`. This file allows for customization of the processing steps, data sources and output files. Each setting within the configuration file is thoroughly documented.

## Color correction

Satellite imagery often appears de-saturated and may not accurately represent true colors, affecting the visual quality of the generated 3D landscape tiles. To address this, the pipeline incorporates three adjustable parameters for image enhancement:

  - Brightness: Adjusts the overall lightness or darkness of the image. A value of 1.0 is unchanged.
  - Contrast: Alters the difference between the darkest and lightest parts of the image, enhancing visual depth. A value of 1.0 is unchanged.
  - Color Enhance: Intensifies the saturation of colors, making the imagery more vivid. A value of 1.0 is unchanged.

The parameter_sweep module is designed to assist in fine-tuning these parameters. It generates a collage of images with varying levels of brightness, contrast, and color enhancement, allowing users to subjectively select the settings that best improve their specific imagery. The default parameters were selected by the author for their particular use case.

![Sample collage](https://github.com/Donitzo/pcd_to_3d_tiles/blob/main/data/sweep/sample_collage.png)

## Pipelines

While the `pcd_to_3d_tiles` module is generic for any point cloud + image combo, the process of reading and subsetting the source files is data-source specific. Each pipeline uses its own script.

### National Land Survey

This pipeline is created for processing geospatial data from the National Land Survey of Finland, which includes laser scanning data in LAS format and satellite or aerial imagery. 

To use data from the National Land Survey of Finland, download laser scanning in the LAS format and image data (such as orthophoto) in any format. The filenames should contain a TM35FIN code which translates to x,y coordinates in meters. The NLS pipeline relies on these filenames to locate the tiles within the world.

Data can be downloaded from the [NLS mapsite](https://www.maanmittauslaitos.fi/en/e-services/mapsite). It's important to attribute the source correctly when using or sharing outputs derived from this data, adhering to the terms of the license. Most data on the map site uses the [National Land Survey open data Attribution CC 4.0 licence](https://www.maanmittauslaitos.fi/en/opendata-licence-cc40).

Downloaded the data and put them into the `data/laser` and `data/orto` directories, or any other directory as long as the configuration points to them.

To initiate the processing of NLS data into 3D tiles, run the following script:

`python pipeline_nls.py`

or

`python pipeline_nls.py --config_file /path/to/config.json`

Ensure that your configuration file is set up correctly, pointing to the directories where your LAS and image data are stored.

## Module description

In addition to the main pipeline, there are a number of helper modules.

### pcd_to_3d_tiles

This module handles conversion from point cloud data/image into a textured 3D mesh. It involves several processing steps:
  - Outlier_removal using nearest neighbor counting
  - Removal of ground points from vegetation using Delaunay triangulation
  - Smoothing vegetation height using nearest neighbor aggregation and percentile height
  - Anisotropic diffusion smoothing of point clouds
  - Model triangulation using 2D Delaunay triangulation
  - Decimation using binary search

The module will save point clouds as PLY models if requested, as well as taking a screenshot of the exported mesh.

## jp2_to_png

Converts JPEG2000 images in the `./data/orto/` directory into PNG format for faster reading and processing.

### parameter_sweep

Manual hyper-parameter optimization with three modes of operation:
  - **Sweep Type 0**: Iterates anisotropic diffusion parameters in `./data/config.json`.
  - **Sweep Type 1**: Iterates target RMSE for decimation in `./data/config.json`.
  - **Sweep Type 2**: Generates a collage of images based on varying image enhancement parameters.

### visualize_mesh

A utility script for visualizing meshes using Open3D. Useful for exporting camera parameters for screenshots.

## Contributing

Contributions/Bug reports to the toolkit are welcome.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
