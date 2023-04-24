import cv2
import numpy as np
import os
import glob
from tqdm import tqdm

# Camera parameters
intrinsic = np.array([
    [1769.60561310104, 0, 1927.08704019384],
    [0, 1763.89532833387, 1064.40054933721],
    [0., 0., 1.]
])
radial_distortion = np.array([
    [-0.244052127306437],
    [0.0597008096110524]
])

# Set distortion coefficients to zero except for radial distortion
distortion = np.zeros((5, 1), dtype=np.float64)
distortion[0:2] = radial_distortion

# Path to the folder containing images to undistort
input_folder = 'preprocessed/scene'
output_folder = 'preprocessed/undistorted_scene'

# Create output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all image files in the folder
image_files = glob.glob(os.path.join(input_folder, '*.jpg')) + glob.glob(os.path.join(input_folder, '*.png'))

# Undistort and save images
for img_file in tqdm(image_files, desc='Undistorting images', unit='image'):
    # Read the image
    img = cv2.imread(img_file)

    # Undistort the image
    undistorted_img = cv2.undistort(img, intrinsic, distortion)

    # Save the undistorted image
    output_img_file = os.path.join(output_folder, os.path.basename(img_file))
    cv2.imwrite(output_img_file, undistorted_img)

print("Undistorted images saved to:", output_folder)
