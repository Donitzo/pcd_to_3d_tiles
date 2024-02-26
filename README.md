# PCD to 3D tiles

This is a simple Python toolkit for transforming point cloud data, specifically in the [LAS format](https://en.wikipedia.org/wiki/LAS_file_format), into textured 3D landscape tiles. Designed with simplicity in mind, the toolkit focuses on generating landscape tiles that are heavily optimized and lightweight.

## Disclaimer

The landscape tiles generated by the toolkit serve as a basic visual representation of the terrain, prioritizing aesthetic appeal over geographical accuracy. As such, the output should not be used for measurements or analysis where exact dimensions and positions are important.

## Installation

Begin by cloning the GitHub repository to your local machine:

`git clone https://github.com/Donitzo/pcd_to_3d_tiles`

and install the requirements in your Python environment of choice:

`pip install -r requirements.txt`

Reasoning behind the requirements:

  - `laspy[lazrs]`: Enables reading LAS and LAZ point cloud files.
  - `open3d`: A 3D library offering comprehensive mesh and point cloud operations for visualization and processing.
  - `pillow`: For reading/writing images.
  - `scipy`: For using spatial models such as Delaunay and cKDTree.
  - `shapely`: Used by trimesh.
  - `tm35fin`: A simple module which can be used to convert coordinates from and to TM35FIN.
  - `trimesh`: Used for reading and writing model files with texture support.
  - `tqdm`: For progress bars.

## Point cloud format

The LAS files should be classified according to the classification standards defined by the [American Society for Photogrammetry and Remote Sensing (ASPRS)](https://www.asprs.org/wp-content/uploads/2019/03/LAS_1_4_r14.pdf).

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

The conversion pipeline processes points classified as Low, Medium, or High Vegetation in certain preprocessing steps. All other classes are treated as ground.

## Configuration

The conversion pipeline's behavior is controlled by a configuration file, by default `data/config.json`. The configuration allows for customization of the processing steps, data sources and output files. Each setting within the configuration file is thoroughly documented.

## Processing steps

### Outlier removal

Outliers in the point cloud are eliminated using local outlier detection. This method identifies and removes points that have fewer neighbors within a specified radius than expected.

### Smoothing vegetation height

Vegetation points are smoothed by adjusting the height of each point to match a percentile of the heights from its neighboring vegetation points, producing a more uniform vegetation layer.

### Filtering ground points

To reduce complexity in the final mesh, ground points located under dense vegetation are removed. This is achieved by triangulating the vegetation from above into a mesh, where the mesh's maximum triangle edge length is constrained. Subsequently, any ground points falling within this mesh (in a 2D perspective) are excluded.

![Point cloud processing](https://github.com/Donitzo/pcd_to_3d_tiles/blob/main/data/sweep/point_cloud_processing.png)

### Anisotropic diffusion

Point cloud smoothing is performed using anisotropic diffusion, which selectively smooths areas while aiming to preserve significant edges within the cloud.

![Anisotropic diffusion collage](https://github.com/Donitzo/pcd_to_3d_tiles/blob/main/data/sweep/sweep_anisotropic_diffusion_collage.png)

### Mesh triangulation

The point cloud is converted into a mesh through Delaunay triangulation, applied from a top-down perspective, to ensure a coherent mesh structure. Far-away corners are added to ensure the mesh covers the entire tile, and the final mesh is cropped back into the tile size.

### Mesh decimation

The mesh undergoes decimation, reducing its complexity by a certain decimation factor. This factor is determined through a binary search that seeks to minimize the Root-mean-square error (RMSE) between the original point cloud and the mesh, adhering to a predefined target RMSE value.

### Color correction

Satellite imagery often appears de-saturated and may not accurately represent true colors. To address this, the conversion pipeline incorporates three adjustable parameters for image enhancement:

  - Brightness: Adjusts the overall lightness or darkness of the image. A value of 1.0 is unchanged.
  - Contrast: Alters the difference between the darkest and lightest parts of the image, enhancing visual depth. A value of 1.0 is unchanged.
  - Color Enhance: Intensifies the saturation of colors, making the imagery more vivid. A value of 1.0 is unchanged.

The `parameter_sweep.py` module is designed to assist in fine-tuning these parameters. It generates a collage of images with varying levels of brightness, contrast, and color enhancement, allowing users to subjectively select the settings that best improve their specific imagery. The default parameters were selected by the author for their particular use case.

![Color correction collage](https://github.com/Donitzo/pcd_to_3d_tiles/blob/main/data/sweep/sample_collage.png)

## Pipelines

While the `pcd_to_3d_tiles.py` module is generic for any point cloud + image combo, the process of reading and subsetting the source files is data-source specific. For this reason, each pipeline uses its own script. For now the only pipeline is `pipeline_nls.py`.

### National Land Survey

This pipeline was created for processing geospatial data from the National Land Survey of Finland, which includes laser scanning data in LAS format and satellite or aerial imagery. 

To use data from the National Land Survey of Finland, download laser scanning data in the LAS format and image data (such as orthophoto) in any format. The filenames should contain a TM35FIN code which translates to x,y coordinates in meters (such as `P3234F3.laz`). The NLS pipeline relies on these filenames to locate the tiles within the world.

Data can be downloaded from the [NLS mapsite](https://www.maanmittauslaitos.fi/en/e-services/mapsite). It's important to attribute the source correctly when using or sharing outputs derived from this data, adhering to the terms of the license. Most data on the map site uses the [National Land Survey open data Attribution CC 4.0 licence](https://www.maanmittauslaitos.fi/en/opendata-licence-cc40).

Download the data and put it into the `data/laser` and `data/orto` directories, or any other directory as long as the configuration file points to them.

To initiate the processing of NLS data into 3D tiles, run the following script:

`python pipeline_nls.py`

or to use another configuration file:

`python pipeline_nls.py --config_file /path/to/config.json`

## Module description

In addition to the main modules, there are a number of small helper modules.

### pcd_to_3d_tiles.py

This module handles conversion from point cloud data/image into a textured 3D mesh. It involves the following steps:
  - Outlier_removal using nearest neighbor counting.
  - Removal of ground points from vegetation using Delaunay triangulation.
  - Smoothing vegetation height using nearest neighbor aggregation and percentile height.
  - Anisotropic diffusion smoothing of point clouds.
  - Model triangulation using 2D Delaunay triangulation.
  - Decimation using binary search to find an optimal decimation factor.

The module will save point clouds as PLY models if requested, as well as taking a screenshot of the exported mesh from a pre-defined camera view.

## jp2_to_png.py

Converts JPEG2000 images in the `./data/orto/` directory into the PNG format for faster reading and processing.

### parameter_sweep.py

Manual parameter optimization with three modes of operation:
  - **Sweep Type 0**: Iterates anisotropic diffusion parameters in `./data/config.json` and outputs models into `./output/sweep_*`.
  - **Sweep Type 1**: Iterates target RMSE for decimation in `./data/config.json` and outputs models into `./output/sweep_*`.
  - **Sweep Type 2**: Generates a collage of images based on varying image enhancement parameters. Reads the image in `./data/sweep/sample.png` and outputs an image into `./output/sample_collage.png`.

### visualize_mesh.py

A utility script for visualizing meshes using Open3D. Useful for exporting camera parameters for screenshots by clicking `p`.

## Contributing

Contributions/Bug reports to the toolkit are welcome.

## Attribution

The samples used in this GitHub repo are based on data from NLS under the open data license mentioned above. Orthophoto images and laser scanning data 0,5 p were used over Vaasa.

## License

This project is licensed under the MIT License.
