import os
import argparse

def create_database(database_path):
    cmd = f"colmap database_create --database_path {database_path}"
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

def model_converter_to_ply(input_path, output_path, output_filename="points3D_undistort.ply"):
    output_file_path = os.path.join(output_path, output_filename)
    cmd = f"colmap model_converter \
            --input_path {input_path} \
            --output_path {output_file_path} \
            --output_type PLY"
    os.system(cmd)

def main(args):
    # Initial or undistorted images
    image_path = "preprocessed/scene" if args.initial else "preprocessed/undistorted_scene"
    data_folder = "colmap/data" if args.initial else "colmap/data_undistort"

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
        output_path=f"{data_folder}/txt",
        output_type="TXT"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run COLMAP with initial or undistorted images.")
    parser.add_argument("--initial", action="store_true", help="Use initial images (default)")
    parser.add_argument("--undistorted", action="store_true", help="Use undistorted images")
    args = parser.parse_args()

    if not (args.initial or args.undistorted):
        args.initial = True

    main(args)
