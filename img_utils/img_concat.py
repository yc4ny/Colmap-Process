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

    # Ensure that all folders have the same number of images
    if len(set(map(len, images_folders))) != 1:
        raise ValueError("The number of images in all folders must be the same")

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
