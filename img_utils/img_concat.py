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
import cv2  
from tqdm import tqdm  

# Define command-line arguments
parser = argparse.ArgumentParser(description="Concatenate images")
parser.add_argument("--folders", nargs='+', help="Paths to the folders containing images")
parser.add_argument("--direction", choices=['horizontal', 'vertical'], default='horizontal', help="Direction of concatenation")
parser.add_argument("--output", help="Path to the output folder")
args = parser.parse_args()

# Define the main function for concatenating images
def concatenate_images(folders, direction, output_folder):
    # Get the sorted image files from all folders
    images_folders = [sorted([f for f in os.listdir(folder) if f.endswith(('.jpg', '.jpeg', '.png'))]) for folder in folders]
    # print(images_folders)
    # # Ensure that all folders have the same number of images
    # if len(set(map(len, images_folders))) != 1:
    #     raise ValueError("The number of images in all folders must be the same")

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Concatenate function based on direction
    concat_func = cv2.hconcat if direction == 'horizontal' else cv2.vconcat

    # Iterate through images and concatenate them
    for i in tqdm(range(len(images_folders[0])), desc="Concating images..."):
        img_paths = [os.path.join(folder, images_folders[j][i]) for j, folder in enumerate(folders)]
        imgs = [cv2.imread(img_path) for img_path in img_paths]

        # Ensure all images have the same width or height based on direction
        if direction == 'horizontal' and len(set(img.shape[0] for img in imgs)) != 1:
            raise ValueError("All images must have the same height for horizontal concatenation")
        elif direction == 'vertical' and len(set(img.shape[1] for img in imgs)) != 1:
            raise ValueError("All images must have the same width for vertical concatenation")

        # Concatenate the images and save the output
        concatenated_img = concat_func(imgs)
        output_image_path = os.path.join(output_folder, images_folders[0][i])
        cv2.imwrite(output_image_path, concatenated_img)

# Call the main function with the command-line arguments
if __name__ == "__main__":
    concatenate_images(args.folders, args.direction, args.output)
