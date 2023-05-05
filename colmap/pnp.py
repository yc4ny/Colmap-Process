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
import subprocess
import argparse

def run_command(command):
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)

def save_output_as_txt(input_path, output_path, output_type):
    os.makedirs(output_path, exist_ok=True)
    cmd = f"colmap model_converter \
            --input_path {input_path} \
            --output_path {output_path} \
            --output_type {output_type}"
    os.system(cmd)

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Preprocessing mp4 files')
    parser.add_argument('--images', help='images.txt output file from colmap', default='data/hand', required=False)
    parser.add_argument('--database', help='colmap database path', default='colmap_data/database.db', required=False)
    parser.add_argument('--existing_reconstruction', help='save path of output pkl file', default='colmap_data/sparse/0', required=False)
    parser.add_argument('--output_path', help='output path of pnp result', default='colmap_data/hand', required=False)
    args = parser.parse_args()

    # Set the paths
    image_path = args.images
    database_path = args.database
    input_path = args.existing_reconstruction
    output_path = args.output_path
    
    # Feature extraction
    feature_extractor_cmd = f"colmap feature_extractor --image_path {image_path} --database_path {database_path} --ImageReader.single_camera_per_folder 1  --ImageReader.camera_model SIMPLE_PINHOLE"
    # Use if need to put camera intrinsic prior:  --ImageReader.camera_params 1929.11,1920,1080 
    run_command(feature_extractor_cmd)
    
    # Feature matching
    exhaustive_matcher_cmd = f"colmap exhaustive_matcher --database_path {database_path}"
    run_command(exhaustive_matcher_cmd)
    
    # Register new images and estimate camera extrinsics
    mapper_cmd = f"colmap image_registrator --database_path {database_path} --input_path {input_path} --output_path {output_path}"
    run_command(mapper_cmd)

    # Save output as txt file
    save_output_as_txt(
        input_path = output_path,
        output_path= output_path,
        output_type="TXT"
    )

if __name__ == "__main__":
    main()
