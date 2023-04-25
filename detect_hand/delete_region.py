import argparse
import cv2
import mediapipe as mp
import os
from tqdm import tqdm 

def isRight(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    index_finger_base = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP]

    if thumb_tip.x < index_finger_base.x:
        return True
    else:
        return False

def process_images(input_folder, output_folder):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0) as hands:

        for img_name in tqdm(sorted(os.listdir(input_folder)), desc = "Detecting hand bbox, blacking out region..."):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(input_folder, img_name)
                image = cv2.imread(img_path)

                if image is None:
                    continue

                height, width, _ = image.shape
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        if isRight(hand_landmarks):
                            x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * width) - 150
                            x_max = image.shape[1]
                            y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * height) - 150
                            y_max = int(max([landmark.y for landmark in hand_landmarks.landmark]) * height) + 150
                        else:
                            x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * width) - 500
                            x_max = int(max([landmark.x for landmark in hand_landmarks.landmark]) * width) + 150
                            y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * height) - 150
                            y_max = image.shape[0]
                        # Draw a black rectangle over the detected hand
                        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 0), -1)
                        cv2.rectangle(image, (1800, 1514), (2800, 2180), (0,0,0), -1)

            # Save the image to the output folder
            output_path = os.path.join(output_folder, "processed_" + img_name)
            cv2.imwrite(output_path, image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect hands and cover them with black rectangles.")
    parser.add_argument("--input", type=str, help="Path to the input folder containing JPG images.", default = "preprocessed/undistorted_left")
    parser.add_argument("--output", type=str, help="Path to the output folder where modified images will be saved.", default = "preprocessed/region_left")
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    process_images(args.input, args.output)

