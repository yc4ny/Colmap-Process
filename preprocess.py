import os
import cv2
import shutil
import argparse
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

parser = argparse.ArgumentParser(description='Preprocessing mp4 files')
parser.add_argument('--scene', help='static scene video file', default='videos/scene.mp4', required=False)
parser.add_argument('--camera_1', help='video file of moving camera in scene', default='videos/left.mp4', required=False)
parser.add_argument('--camera_2', help='video file of moving camera in scene', default='videos/right.mp4', required=False)
parser.add_argument('--output', help='output dir of processed frames', default='preprocessed', required=False)
args = parser.parse_args()

def extract_frames(video, output, path, is_scene=False):
    basename = os.path.basename(path)
    basename, _ = os.path.splitext(basename)
    save_dir = os.path.join(output, basename)

    os.makedirs(save_dir, exist_ok=True)

    frame_num = 0

    with tqdm(total=int(video.get(cv2.CAP_PROP_FRAME_COUNT)), desc=f"Extracting frames from {basename}...") as pbar:
        while True:
            ret, frame = video.read()

            if not ret:
                break
            output_path = os.path.join(save_dir, f"{frame_num:05d}.jpg")
            cv2.imwrite(output_path, frame)
            frame_num += 1
            pbar.update(1)

    if is_scene:
        sample_scene_frames(output, save_dir, basename)

def sample_scene_frames(output, save_dir, basename):
    image_files = sorted([f for f in os.listdir(save_dir) if f.endswith('.jpg')])
    sampled_dir = os.path.join(output, f"sampled_{basename}")

    os.makedirs(sampled_dir, exist_ok=True)

    for i, image_file in tqdm(enumerate(image_files), total=len(image_files), desc="Sampling images..."):
        if (i + 1) % 5 == 0:
            shutil.copy(os.path.join(save_dir, image_file), sampled_dir)

    print(f"{len(os.listdir(sampled_dir))} scene images sampled out of {len(os.listdir(save_dir))} original images.")
    shutil.rmtree(save_dir)

def main():
    print("-------------------------Preprocessing video data-------------------------")
    start = time.time()
    os.makedirs(args.output, exist_ok=True)

    scene_vid = cv2.VideoCapture(args.scene)
    camera1_vid = cv2.VideoCapture(args.camera_1)
    camera2_vid = cv2.VideoCapture(args.camera_2)

    tasks = [
        (scene_vid, args.output, args.scene, True),
        (camera1_vid, args.output, args.camera_1, False),
        (camera2_vid, args.output, args.camera_2, False)
    ]

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_frames, *task) for task in tasks]

        for future in as_completed(futures):
            future.result()

    end = time.time()
    print(f"Preprocess took {(end-start)/60:.5f} minutes")

if __name__ == "__main__":
    main()
