import os
import cv2
import shutil
import argparse
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Argument parser setup
parser = argparse.ArgumentParser(description='Preprocessing mp4 files')
parser.add_argument('--scene', help='static scene video file', default='videos/scene.mp4', required=False)
parser.add_argument('--camera_1', help='video file of moving camera in scene', default='videos/hand.mp4', required=False)
parser.add_argument('--camera_2', help='video file of moving camera in scene', default='videos/right.MP4', required=False)
parser.add_argument('--output', help='output dir of processed frames', default='preprocessed', required=False)
args = parser.parse_args()

# JPEG quality setting
JPEG_QUALITY = 100

def extract_frames(video, output, path, is_scene=False):
    # Get the base file name and remove the file extension
    basename = os.path.basename(path)
    basename, _ = os.path.splitext(basename)

    # Create a directory for saving the extracted frames
    save_dir = os.path.join(output, basename)
    os.makedirs(save_dir, exist_ok=True)

    frame_num = 0

    # Extract frames from the video and save them as JPEG images
    with tqdm(total=int(video.get(cv2.CAP_PROP_FRAME_COUNT)), desc=f"Extracting frames from {basename}...") as pbar:
        while True:
            ret, frame = video.read()

            if not ret:
                break
            output_path = os.path.join(save_dir, f"{basename}_{frame_num:05d}.jpg")
            cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            frame_num += 1
            pbar.update(1)

    # If it's a scene video, sample the frames
    if is_scene:
        sample_scene_frames(output, save_dir, basename)

def sample_scene_frames(output, save_dir, basename):
    # Get a list of all JPEG image files in the save directory
    image_files = sorted([f for f in os.listdir(save_dir) if f.endswith('.jpg')])

    # Create a directory for saving the sampled frames
    sampled_dir = os.path.join(output, f"sampled_{basename}")
    os.makedirs(sampled_dir, exist_ok=True)

    # Copy every 5th frame to the sampled directory
    for i, image_file in tqdm(enumerate(image_files), total=len(image_files), desc="Sampling images..."):
        if (i + 1) % 5 == 0:
            shutil.copy(os.path.join(save_dir, image_file), sampled_dir)

    # Print the number of sampled images and clean up the original save directory
    print(f"{len(os.listdir(sampled_dir))} scene images sampled out of {len(os.listdir(save_dir))} original images.")
    shutil.rmtree(save_dir)

def main():
    # Print the start message for preprocessing
    print("-------------------------Preprocessing video data-------------------------")
    
    # Record the starting time
    start = time.time()
    
    # Create the output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Open the video files using OpenCV
    scene_vid = cv2.VideoCapture(args.scene)
    camera1_vid = cv2.VideoCapture(args.camera_1)
    camera2_vid = cv2.VideoCapture(args.camera_2)

    # Create a list of tasks for each video file
    tasks = [
        (scene_vid, args.output, args.scene, True),
        # (camera1_vid, args.output, args.camera_1, False),
        # (camera2_vid, args.output, args.camera_2, False)
    ]

    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor() as executor:
        # Submit the tasks to the executor and store the futures
        futures = [executor.submit(extract_frames, *task) for task in tasks]

        # Wait for each future to complete
        for future in as_completed(futures):
            future.result()

    # Record the end time
    end = time.time()
    
    # Calculate and print the elapsed time in minutes
    print(f"Preprocess took {(end-start)/60:.5f} minutes")

if __name__ == "__main__":
    main()
