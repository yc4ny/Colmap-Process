import os
import subprocess

def run_command(command):
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)

def main():
    # Set the paths
    image_path = "preprocessed/hand"
    existing_reconstruction_path = "colmap_data"
    database_path = os.path.join(existing_reconstruction_path, "database.db")
    input_path = os.path.join(existing_reconstruction_path, "sparse/0")
    output_path = os.path.join(existing_reconstruction_path, "hand")
    
    # Feature extraction
    feature_extractor_cmd = f"colmap feature_extractor --image_path {image_path} --database_path {database_path} --ImageReader.single_camera_per_folder 1  --ImageReader.camera_model SIMPLE_RADIAL --ImageReader.camera_params 1136.88,360,640,0.0234776"
    run_command(feature_extractor_cmd)
    
    # Feature matching
    exhaustive_matcher_cmd = f"colmap exhaustive_matcher --database_path {database_path}"
    run_command(exhaustive_matcher_cmd)
    
    # Register new images and estimate camera extrinsics
    mapper_cmd = f"colmap image_registrator --database_path {database_path} --input_path {input_path} --output_path {output_path}"
    run_command(mapper_cmd)

if __name__ == "__main__":
    main()
