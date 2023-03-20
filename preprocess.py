import os 
import cv2 
import shutil 
import argparse

parser = argparse.ArgumentParser(description = 'Preprocessing *mp4 files')
parser.add_argument('--scene', help = 'static scene video file', default ='videos/scene.mp4', required = False)
parser.add_argument('--camera_1', help = 'video file of moving camera in scene', default = 'videos/left_3.mp4', required = False)
parser.add_argument('--camera_2', help = 'video file of moving camera in scene', default = 'videos/right_3.mp4', required = False)
parser.add_argument('--output', help = 'output dir of processed frames', default = 'preprocessed', required = False)
args = parser.parse_args()

def extractFrames(video, output, path):
    basename = os.path.basename(path)
    basename, _ = os.path.splitext(basename) # e.g) scene
    save_dir = output + "/" + basename
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    frame_num = 0 

    while True:
        ret, frame = video.read()

        if not ret:
            break 
        
        output_path = os.path.join(save_dir, f"{frame_num:05d}.jpg")
        cv2.imwrite(output_path, frame)
        frame_num += 1




def main(): 

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    scene_vid = cv2.VideoCapture(args.scene)
    camera1_vid = cv2.VideoCapture(args.camera_1)
    camera2_vid = cv2.VideoCapture(args.camera_2)
    extractFrames(scene_vid, args.output, args.scene)

if __name__ == "__main__":
    main()
