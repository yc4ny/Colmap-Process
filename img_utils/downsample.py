import cv2
import argparse
import os

def extract_and_resize(video_path, output_folder, resize_factor=0.25):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Couldn't open the video file {video_path}")
        return

    frame_number = 0
    while True:
        ret, frame = cap.read()

        # If frame is read correctly, it will be True; otherwise, break from the loop
        if not ret:
            break

        # Resize the frame
        resized_frame = cv2.resize(frame, (int(frame.shape[1] * resize_factor), int(frame.shape[0] * resize_factor)))

        # Save the resized frame
        frame_filename = os.path.join(output_folder, f"frame_{frame_number:04d}.jpg")
        cv2.imwrite(frame_filename, resized_frame)

        frame_number += 1

    cap.release()
    print(f"Processed {frame_number} frames and saved to {output_folder}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from an MP4 video and resize them.")
    parser.add_argument("--video_path", type=str, help="Path to the input MP4 video file.")
    parser.add_argument("--output_folder", type=str, default="output_frames", help="Folder to save the extracted frames.")
    
    args = parser.parse_args()

    extract_and_resize(args.video_path, args.output_folder)
