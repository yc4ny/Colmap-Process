# MIT License
#
# Copyright (c) 2023 Yonwoo Choi, Seoul National University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import argparse

def create_database(database_path):
    cmd = f"colmap database_creator --database_path {database_path}"
    os.system(cmd)

def feature_extraction(database_path, image_path, single_camera, camera_model):
    cmd = f"colmap feature_extractor \
            --database_path {database_path} \
            --image_path {image_path} \
            --ImageReader.single_camera {single_camera} \
            --ImageReader.camera_model {camera_model}"
            # --ImageReader.camera_params 1910.2,1920,1080" 
    os.system(cmd)

def feature_matching(database_path):
    cmd = f"colmap exhaustive_matcher --database_path {database_path}"
    os.system(cmd)

def sparse_reconstruction(database_path, image_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    cmd = f"colmap mapper \
            --database_path {database_path} \
            --image_path {image_path} \
            --output_path {output_path}"
    os.system(cmd)

def save_output_as_txt(input_path, output_path, output_type):
    os.makedirs(output_path, exist_ok=True)
    cmd = f"colmap model_converter \
            --input_path {input_path} \
            --output_path {output_path} \
            --output_type {output_type}"
    os.system(cmd)

def save_output_as_ply(input_path, output_path, output_filename="points.ply"):
    output_file_path = os.path.join(output_path, output_filename)
    cmd = f"colmap model_converter \
            --input_path {input_path} \
            --output_path {output_file_path} \
            --output_type PLY"
    os.system(cmd)

def main(args):
    data_folder = args.colmap_data
    image_path = args.image_path

    os.makedirs(data_folder, exist_ok=True)

    # Create Database
    create_database(database_path=f"{data_folder}/database.db")

    # Feature Extraction
    feature_extraction(
        database_path=f"{data_folder}/database.db",
        image_path=image_path,
        single_camera=1,
        camera_model= args.camera_model
    )

    # Feature Matching
    feature_matching(database_path=f"{data_folder}/database.db")

    # Sparse Reconstruction
    sparse_reconstruction(
        database_path=f"{data_folder}/database.db",
        image_path=image_path,
        output_path=f"{data_folder}/sparse"
    )

    # Save output as txt file
    save_output_as_txt(
        input_path=f"{data_folder}/sparse/0",
        output_path=f"{data_folder}/sparse/0",
        output_type="TXT"
    )

    save_output_as_ply(
        input_path=f"{data_folder}/sparse/0",
        output_path=f"{data_folder}/sparse/0",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run COLMAP with initial or undistorted images.")
    parser.add_argument("--image_path", action="store_true", help="Image paths", default = "preprocessed/undistort_opencv_scene")
    parser.add_argument("--colmap_data",action="store_true", help="Path to colmap data", default = "colmap_data")
    parser.add_argument("--camera_model",action="store_true", help="Type of camera model", default = "SIMPLE_PINHOLE")
    args = parser.parse_args()

    main(args)
