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
    parser.add_argument('--images', help='Image path', default='preprocessed/undistort_right', required=False)
    parser.add_argument('--database', help='colmap database path', default='colmap_data/database.db', required=False)
    parser.add_argument('--existing_reconstruction', help='existing bin file path', default='colmap_data/left', required=False)
    parser.add_argument('--output_path', help='output path of pnp result', default='colmap_data/right', required=False)
    args = parser.parse_args()

    # Set the paths
    image_path = args.images
    database_path = args.database
    input_path = args.existing_reconstruction
    output_path = args.output_path
    
    # Feature extraction
    feature_extractor_cmd = f"colmap feature_extractor --image_path {image_path} --database_path {database_path} --ImageReader.single_camera_per_folder 1  --ImageReader.camera_model SIMPLE_PINHOLE --ImageReader.camera_params 1920.84,1920,1080"
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
