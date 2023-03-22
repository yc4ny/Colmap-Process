import os
import cv2
import numpy as np
import argparse
from collections import OrderedDict
from tqdm import tqdm

# Argument parser
parser = argparse.ArgumentParser(description='Visualzing reprojection')
parser.add_argument('--cameras', help='cameras.txt file', default='colmap/data_undistort/txt/cameras.txt', required=False)
parser.add_argument('--points', help='points3D.txt file', default='colmap/data_undistort/txt/points3D.txt', required=False)
parser.add_argument('--images', help='images.txt file', default='colmap/data_undistort/txt/images.txt', required=False)
args = parser.parse_args()

# Read camera intrinsic parameters from the file
def read_camera(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the line containing the camera information
    for line in lines:
        if line.strip() and not line.startswith('#'):
            camera_line = line
            break

    _, _, width, height, *params = camera_line.split()
    fx, fy, cx, cy = map(float, params[:4])

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    return K

# Read 3D points,color information from file
def read_points(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()[3:]  # Skip the header lines

    points = []
    colors = []
    for line in lines:
        if line.strip():
            id, x, y, z, r, g, b, _ = line.split(None, 7)
            points.append([float(x), float(y), float(z)])
            colors.append([int(r), int(g), int(b)])

    return np.array(points), np.array(colors)

# Read image extrinsic parameters and 2D keypoints from the file
def read_images(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    extrinsic_parameters = {}
    keypoints = {}
    for idx, line in enumerate(lines):
        if line.startswith('#') or line.strip() == '':
            continue
        if idx % 2 == 0:
            line_parts = line.strip().split()
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name = line_parts
            rotation = quaternion_to_rotation_matrix(float(qw), float(qx), float(qy), float(qz))
            translation = np.array([float(tx), float(ty), float(tz)]).reshape(3, 1)
            extrinsic = np.hstack((rotation, translation))
            extrinsic_parameters[name] = extrinsic
        else:
            coords = line.strip().split()
            coords = [float(x) for x in coords]
            coords = np.array(coords).reshape(-1, 3)[:, :2]
            keypoints[name] = coords

    sorted_names = sorted(extrinsic_parameters.keys())
    extrinsics_array = np.stack([extrinsic_parameters[name] for name in sorted_names], axis=0)
    keypoints_array = [keypoints[name] for name in sorted_names]

    return extrinsics_array, keypoints_array

# Convert quaternion to rotation matrix
def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    R = np.zeros((3, 3))
    R[0, 0] = 1 - 2 * qy ** 2 - 2 * qz ** 2
    R[0, 1] = 2 * qx * qy - 2 * qz * qw
    R[0, 2] = 2 * qx * qz + 2 * qy * qw
    R[1, 0] = 2 * qx * qy + 2 * qz * qw
    R[1, 1] = 1 - 2 * qx ** 2 - 2 * qz ** 2
    R[1, 2] = 2 * qy * qz - 2 * qx * qw
    R[2, 0] = 2 * qx * qz - 2 * qy * qw
    R[2, 1] = 2 * qy * qz + 2 * qx * qw
    R[2, 2] = 1 - 2 * qx ** 2 - 2 * qy ** 2

    return R

# Reproject 3D points on images and visualize the reprojection error
def reproject(K, Rt, points3D, colors, points2D, input_folder, output_folder):
    # Get the sorted image files from the input folder
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])
   
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize total error and total points counters
    total_error = 0
    total_points = 0

    # Iterate through images and reproject 3D points
    for idx, image_file in tqdm(enumerate(image_files), total=len(image_files), desc="Reprojecting images..."):
        input_image_path = os.path.join(input_folder, image_file)
        # Read the input image
        img = cv2.imread(input_image_path)
        # Extract rotation and translation from the current image's extrinsic parameters
        R = Rt[idx, :3, :3]
        t = Rt[idx, :3, 3]
        # Initialize frame error and frame points counters
        frame_error = 0
        frame_points = 0
        # Project 3D points on the image and calculate the reprojection error
        for i, point in enumerate(points3D):
            projected_point = K @ (R @ point + t)
            u = projected_point[0] / projected_point[2]
            v = projected_point[1] / projected_point[2]

            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                color = tuple(reversed(colors[i]))  # Convert RGB to BGR for OpenCV
                # cv2.circle(img, (int(u), int(v)), 3, tuple((int(color[0]), int(color[1]), int(color[2]))), -1)
                cv2.circle(img, (int(u), int(v)), 2,(0,255,0), -1)
                if i < len(points2D[idx]):
                    error = np.linalg.norm(points2D[idx][i] - np.array([u, v]))
                    frame_error += error
                    frame_points += 1
        # Update total error and total points counters
        total_error += frame_error
        total_points += frame_points
        # Calculate the average error so far
        avg_error_so_far = total_error / total_points

        # Get image dimensions
        height, width, _ = img.shape

        # Calculate font scale factor and text positions based on image dimensions
        font_scale = min(width, height) / 800  # Adjust the denominator to change the relative font size
        text_pos1 = (int(0.05 * width), int(0.08 * height))
        text_pos2 = (int(0.05 * width), int(0.16 * height))

        # cv2.putText(img, f'Frame reprojection error: {frame_error/frame_points:.2f}', text_pos1, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 5, cv2.LINE_AA)
        # cv2.putText(img, f'Avg. error so far: {avg_error_so_far:.2f}', text_pos2, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 5, cv2.LINE_AA)
        output_image_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_image_path, img)
    
    print("------------------Average Reprojection Error: " + str(avg_error_so_far) + " ------------------")

def main():
    K = read_camera(args.cameras)
    points3D, color = read_points(args.points)
    Rt, keypoints2d = read_images(args.images)
    reproject(K,Rt,points3D, color, keypoints2d, "preprocessed/undistorted_scene", "preprocessed/reproject_undistorted_scene")


if __name__ == "__main__":
    main()