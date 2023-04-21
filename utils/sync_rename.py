import os
import re

def rename_files_in_folder(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)

    # Filter image files with the pattern: left_XXXXX.jpg
    image_files = [f for f in files if re.match(r'right_\d{5}\.jpg', f)]

    # Sort image files
    image_files.sort()

    # Iterate over the image files and rename them
    for index, image_file in enumerate(image_files, start=1):
        old_path = os.path.join(folder_path, image_file)
        new_name = f"right_{index:05d}.jpg"
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed {old_path} to {new_path}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python rename_images.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    rename_files_in_folder(folder_path)
