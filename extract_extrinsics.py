import re
import numpy as np
import quaternion
import pickle
import argparse

def extract_extrinsics(images_txt, base):
    extrinsics = {}

    with open(images_txt, 'r') as file:
        lines = file.readlines()

    for i in range(len(lines)):
        line = lines[i].strip()

        if not line or line.startswith('#'):
            continue

        if re.match(f'{base}_\d{{5}}\.jpg', line.split()[-1]):
            image_data = line.split()
            qw, qx, qy, qz = map(float, image_data[1:5])
            tx, ty, tz = map(float, image_data[5:8])
            
            rotation = quaternion.as_rotation_matrix(quaternion.quaternion(qw, qx, qy, qz))
            translation = np.array([tx, ty, tz]).reshape(3, 1)

            extrinsic_matrix = np.hstack((rotation, translation))
            extrinsics[line.split()[-1]] = extrinsic_matrix

    return extrinsics

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Preprocessing mp4 files')
    parser.add_argument('--images_txt', help='images.txt output file from colmap', default='data/desk/colmap_data/left/images.txt', required=False)
    parser.add_argument('--output', help='save directory of output pkl file', default='left_extrinsic.pkl', required=False)
    parser.add_argument('--base', help='base name of the frames taken by camera', default='left', required=False)
    args = parser.parse_args()

    extrinsics = extract_extrinsics(args.images_txt, args.base)

    with open(args.output, 'wb') as file:
        pickle.dump(extrinsics, file)

if __name__ == "__main__":
    main()