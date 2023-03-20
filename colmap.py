import os 

def main():
    # Create Database 
    cmd = "colmap database_create \
    --database_path colmap_data/database.db"
    os.system(cmd)

    # Feature Extraction
    cmd = "colmap feature_extractor \
    --database_path colmap_data/database.db \
    --image_path preprocessed/sampled_scene \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model OPENCV"
    os.system(cmd)

    # Feature Matching
    cmd = "colmap exhaustive_matcher \
    --database_path colmap_data/database.db"
    os.system(cmd)

    # Sparse Reconstruction
    cmd = "mkdir colmap_data/sparse"
    os.system(cmd)
    cmd = "colmap mapper \
    --database_path colmap_data/database.db \
    --image_path preprocessed/sampled_scene \
    --output_path colmap_data/sparse"
    os.system(cmd)

if __name__ == "__main__":
    main()
