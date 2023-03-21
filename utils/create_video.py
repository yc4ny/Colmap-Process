import cv2
import os
import sys
import glob2
from natsort import natsorted
from tqdm import tqdm 

# python create_video.py /path/to/your/folder

def create_video_from_images(folder_path, output_file, fps=2):
    # Get all JPG images in the folder
    image_files = natsorted(glob2.glob(os.path.join(folder_path, "*.jpg")))

    # Read the first image to get the dimensions
    first_image = cv2.imread(image_files[0])
    height, width, layers = first_image.shape

    # Create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Add images to the video
    for image_file in tqdm(image_files):
        image = cv2.imread(image_file)
        video_writer.write(image)

    # Release the video writer
    video_writer.release()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_video.py [input_folder_path]")
        sys.exit(1)

    input_folder_path = sys.argv[1]
    folder_name = os.path.basename(os.path.normpath(input_folder_path))
    output_file = f"{folder_name}.mp4"

    create_video_from_images(input_folder_path, output_file)
    print(f"Video created: {output_file}")
