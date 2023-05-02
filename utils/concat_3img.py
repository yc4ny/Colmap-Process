import cv2
import os

# Set the input and output directories
input_dirs = ['preprocessed/reproject_opencv_left', 'preprocessed/reproject_opencv_head', 'preprocessed/reproject_opencv_right']
output_dir = 'preprocessed/concat_reproject'

# Get a list of all image files in the input directories
image_files = []
for input_dir in input_dirs:
    image_files.append([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png')])

# # Check that all folders have the same number of images
num_images = len(image_files[0])
# if not all(len(f) == num_images for f in image_files):
#     raise ValueError('The input folders do not have the same number of images')

# Loop over all images and concatenate them horizontally
for i in range(num_images):
    # Open the three input images
    img1 = cv2.imread(image_files[0][i])
    img2 = cv2.imread(image_files[1][i])
    img3 = cv2.imread(image_files[2][i])

    # Concatenate the images horizontally
    concat_img = cv2.hconcat([img1, img2, img3])

    # Save the concatenated image to the output directory
    output_path = os.path.join(output_dir, f'{i+1}.jpg')
    cv2.imwrite(output_path, concat_img)
