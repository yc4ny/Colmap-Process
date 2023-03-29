import os
import argparse
import cv2
from tqdm import tqdm 

parser = argparse.ArgumentParser(description="Concatenate images vertically")
parser.add_argument("--f1", help="Path to the first folder containing images")
parser.add_argument("--f2", help="Path to the second folder containing images")
parser.add_argument("--output", help="Path to the output folder", default="output")
args = parser.parse_args()

def concatenate_images(folder1, folder2, output_folder):
    # Get the sorted image files from both folders
    images_folder1 = sorted([f for f in os.listdir(folder1) if f.endswith(('.jpg', '.jpeg', '.png'))])
    images_folder2 = sorted([f for f in os.listdir(folder2) if f.endswith(('.jpg', '.jpeg', '.png'))])

    if len(images_folder1) != len(images_folder2):
        raise ValueError("The number of images in both folders must be the same")

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through images and concatenate them vertically
    for img_file1, img_file2 in tqdm(zip(images_folder1, images_folder2), total=len(images_folder1), desc="Concating images..."):
        img_path1 = os.path.join(folder1, img_file1)
        img_path2 = os.path.join(folder2, img_file2)

        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)

        if img1.shape[1] != img2.shape[1]:
            raise ValueError(f"Images {img_file1} and {img_file2} must have the same width")

        concatenated_img = cv2.vconcat([img1, img2])
        output_image_path = os.path.join(output_folder, img_file1)
        cv2.imwrite(output_image_path, concatenated_img)

if __name__ == "__main__":
    concatenate_images(args.f1, args.f2, args.output)
