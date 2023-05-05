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
import re  
import argparse 

# Define command-line arguments
parser = argparse.ArgumentParser(description="Rename image files in a folder")
parser.add_argument("--folder_path", help="Path to the folder containing image files")
parser.add_argument("--base_name", help="Base name for the new file names", default="image")
args = parser.parse_args()

# Define the main function for renaming files
def rename_files_in_folder(folder_path, base_name):
    # List all files in the folder
    files = os.listdir(folder_path)

    # Filter image files with the pattern: base_name_XXXXX.jpg
    pattern = re.compile(f"{base_name}_\d{{5}}\.jpg")
    image_files = [f for f in files if pattern.match(f)]

    # Sort image files
    image_files.sort()

    # Iterate over the image files and rename them
    for index, image_file in enumerate(image_files, start=1):
        old_path = os.path.join(folder_path, image_file)
        new_name = f"{base_name}_{index:05d}.jpg"
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed {old_path} to {new_path}")

# Call the main function with the command-line arguments
if __name__ == "__main__":
    # Parse command-line arguments
    folder_path = args.folder_path
    base_name = args.base_name

    # Call the function to rename files
    rename_files_in_folder(folder_path, base_name)
