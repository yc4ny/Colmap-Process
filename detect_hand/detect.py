import cv2
import mediapipe as mp
import os
import json
from tqdm import tqdm

def detect_hand_joints(input_folder, output_folder, output_images_folder):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(output_images_folder):
        os.makedirs(output_images_folder)

    filenames = [filename for filename in os.listdir(input_folder) if filename.endswith(".jpg")]

    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.1) as hands:
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

                    if handedness == "Right":
                        hand_type = "left"
                    else:
                        hand_type = "right"

                    joint_coordinates = []
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                        joint_coordinates.append((x, y))

                    hand_joints[hand_type].append(joint_coordinates)

                    # Draw hand landmarks on the image
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            with open(output_path, "w") as outfile:
                json.dump(hand_joints, outfile)

            # Save the image with hand landmarks
            cv2.imwrite(output_image_path, image)


if __name__ == "__main__":
    input_folder = "preprocessed/left"
    output_folder = "detect_hand/left"
    output_images_folder = "preprocessed/left_joints"
    detect_hand_joints(input_folder, output_folder, output_images_folder)
