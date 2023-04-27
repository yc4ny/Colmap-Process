import os
import re
import numpy as np
import quaternion
import pickle

def extract_extrinsics(images_txt):
    extrinsics = {}

    with open(images_txt, 'r') as file:
        lines = file.readlines()

    for i in range(len(lines)):
        line = lines[i].strip()

        if not line or line.startswith('#'):
            continue

        if re.match(r'hand_\d{5}\.jpg', line.split()[-1]):
            image_data = line.split()
            qw, qx, qy, qz = map(float, image_data[1:5])
            tx, ty, tz = map(float, image_data[5:8])
            
            rotation = quaternion.as_rotation_matrix(quaternion.quaternion(qw, qx, qy, qz))
            translation = np.array([tx, ty, tz]).reshape(3, 1)

            extrinsic_matrix = np.hstack((rotation, translation))
            extrinsics[line.split()[-1]] = extrinsic_matrix

    return extrinsics

images_txt_path = 'colmap_data/hand/images.txt'
extrinsics = extract_extrinsics(images_txt_path)

with open('extrinsics.pkl', 'wb') as file:
    pickle.dump(extrinsics, file)

print(f"Saved extrinsics for {len(extrinsics)} images in extrinsics.pkl")