# MIT License
#
# Copyright (c) 2023 Yonwoo Choi, Seoul National University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import cv2
import mediapipe as mp
import os
import json
from tqdm import tqdm
import argparse

def detect_hand_joints(input_folder, output_folder, output_images_folder):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(output_images_folder):
        os.makedirs(output_images_folder)

    filenames = [filename for filename in os.listdir(input_folder) if filename.endswith(".jpg")]

    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
        for filename in tqdm(filenames, desc="Processing images"):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_joints.json")
            output_image_path = os.path.join(output_images_folder, filename)

            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = hands.process(image_rgb)

            hand_joints = {"left": [], "right": []}

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    handedness = results.multi_handedness[results.multi_hand_landmarks.index(hand_landmarks)].classification[0].label

                    # Skip if handedness is not Left
                    if handedness == "Left":
                        continue

                    hand_type = "right"

                    joint_coordinates = []
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                        joint_coordinates.append((x, y))

                    hand_joints[hand_type].append(joint_coordinates)

                    # Draw hand landmarks on the image
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Break the loop after processing the first left hand
                    break

            with open(output_path, "w") as outfile:
                json.dump(hand_joints, outfile)

            # Save the image with hand landmarks
            cv2.imwrite(output_image_path, image)


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Preprocessing mp4 files')
    parser.add_argument('--input', help='directory of folder with hand image frames', default='data/imgs', required=False)
    parser.add_argument('--output_json', help='output directory of json file containing 2d hand joints', default='mediapipe/detected_joints', required=False)
    parser.add_argument('--output_img', help='base name of the frames taken by camera', default='mediapipe/joint_img', required=False)
    args = parser.parse_args()
    detect_hand_joints(args.input, args.output_json, args.output_img)
