import os 
import cv2 
import shutil 
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser(description = 'Preprocessing *mp4 files')
parser.add_argument('--scene', help = 'static scene video file', default ='videos/scene.mp4', required = False)
parser.add_argument('--camera_1', help = 'video file of moving camera in scene', default = 'videos/left_3.mp4', required = False)
parser.add_argument('--camera_2', help = 'video file of moving camera in scene', default = 'videos/right_3.mp4', required = False)
parser.add_argument('--output', help = 'output dir of processed frames', default = 'preprocessed', required = False)
args = parser.parse_args()

def extractSceneFrames(video, output, path):
    basename = os.path.basename(path)
    basename, _ = os.path.splitext(basename) # e.g) scene
    save_dir = output + "/" + basename
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    frame_num = 0 

    # Use tqdm to show a progress bar while reading the frames
    with tqdm(total=int(video.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
        while True:
            ret, frame = video.read()

            if not ret:
                break 
            output_path = os.path.join(save_dir, f"{frame_num:05d}.jpg")
            cv2.imwrite(output_path, frame)
            frame_num += 1
            
            # Update the progress bar
            pbar.update(1)
    
    image_files = [f for f in os.listdir(save_dir) if f.endswith('.jpg')]

    for i, image_file in enumerate(image_files):
        if (i)%5 == 0:
            shutil.copy(os.path.join(save_dir))


def main(): 

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    scene_vid = cv2.VideoCapture(args.scene)
    camera1_vid = cv2.VideoCapture(args.camera_1)
    camera2_vid = cv2.VideoCapture(args.camera_2)
    extractSceneFrames(scene_vid, args.output, args.scene)

if __name__ == "__main__":
    main()
