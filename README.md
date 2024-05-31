# Colmap Tools

This repository contains a collection of Python scripts designed to automate various tasks using COLMAP, an open-source Structure-from-Motion (SfM) and Multi-View Stereo (MVS) software. These tools facilitate processes ranging from 3D reconstruction to camera pose estimation and 3D point reprojection onto 2D images.

## Overview of Scripts

#### run_sfm.py

Automates the full pipeline for 3D reconstruction using COLMAP, including database creation, feature extraction, feature matching, sparse reconstruction, and exporting the output.

#### pnp.py

Performs Perspective-n-Point (PnP) problem solving to estimate camera poses with an existing 3D model.

#### load_txt.py

Provides utility functions for loading camera parameters, image data, and 3D point data from COLMAP's text output files.

#### extract_extrinsics.py

Extracts camera extrinsic parameters from a COLMAP `images.txt` file for further analysis or processing.

#### reproject.py

Demonstrates how to reproject 3D points back onto 2D image planes using camera intrinsic and extrinsic parameters, including handling of quaternion to rotation matrix conversion and point distortion.

## Installation

I have tested this code on Python 1.7.1.

To install the required Python packages, run the following command in your terminal:

```bash
pip install -r requirements.txt
```

Make sure you have COLMAP installed on your system. For installation instructions, refer to the [official COLMAP documentation](https://colmap.github.io/install.html).

## Usage

Each script is designed to be run from the command line with specific arguments. Below are examples for each script:

#### run_sfm.py

```bash
python run_sfm.py --image_path <path_to_images> --colmap_data <path_to_colmap_data> --camera_model <camera_model>
```

#### pnp.py

```bash
python pnp.py --images <path_to_images> --database <path_to_colmap_database> --existing_reconstruction <path_to_existing_model> --output_path <output_path>
```

#### extract_extrinsics.py

```bash
python extract_extrinsics.py --images_txt <path_to_images_txt> --output <output_pickle_file> --base <base_name>
```

#### reproject.py

```bash
python reproject.py --images <path_to_images> --colmap_output <path_to_colmap_output> --camera_id <camera_id> --output <output_path>
```


## Contributing

Contributions are welcome! If you have suggestions for improvements or bug fixes, please feel free to submit a pull request or open an issue.

## License

This project is open source and available under the [MIT License](LICENSE).
