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

def save_output_as_ply(input_path, output_path, output_filename="points3D.ply"):
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
        camera_model="OPENCV"
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
        output_path=f"{data_folder}/sparse/0/",
        output_type="TXT"
    )

    save_output_as_ply(
        input_path=f"{data_folder}/sparse/0",
        output_path=f"{data_folder}/sparse/0/",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run COLMAP with initial or undistorted images.")
    parser.add_argument("--image_path", action="store_true", help="Image paths", default = "preprocessed/scene")
    parser.add_argument("--colmap_data",action="store_true", help="Path to colmap data", default = "colmap_data")
    args = parser.parse_args()

    main(args)
