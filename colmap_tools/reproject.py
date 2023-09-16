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

import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from load_txt import read_cameras_txt, read_images_txt, read_points3D_txt

# Conversion of quaternion to rotation matrix
def quaternion_to_rotation_matrix(quaternion):
    qw, qx, qy, qz = quaternion
    # Calculating the rotation matrix
    rot_matrix = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return rot_matrix

# Distorting points: Colmap output -> Parameters for distorting points to scene
def distort_points(points, k1, k2, p1, p2, fx, fy, cx, cy):
    distorted_points = []
    for point in points:
        # Normalizing the points
        x, y = (point - np.array([cx, cy])) / np.array([fx, fy])
        # Calculating the radial distortion
        r2 = x**2 + y**2
        # Applying the distortion
        x_distorted = x * (1 + k1 * r2 + k2 * r2**2) + 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
        y_distorted = y * (1 + k1 * r2 + k2 * r2**2) + p1 * (r2 + 2 * y**2) + 2 * p2 * x * y
        distorted_points.append([x_distorted * fx + cx, y_distorted * fy + cy])
    return np.array(distorted_points)

# Reprojecting 3d points of a simple pinhole camera model
def reproject_3d_points(images_folder, images_data, points3D, camera_params, camera_id, output):
    fx, cx, cy = camera_params
    # Constructing the camera intrinsic matrix
    intrinsics = np.array([
        [fx, 0, cx],
        [0, fx, cy],
        [0, 0, 1]
    ])

    for img_data in tqdm(images_data, desc="Processing frames"):
        if img_data['camera_id'] != camera_id:
            continue
        img_path = images_folder / img_data['name']
        img = cv2.imread(str(img_path))
        img_shape = img.shape[:2]
        # Obtainting R,t extrinsics parameters
        rot_matrix = quaternion_to_rotation_matrix(img_data['quaternion'])
        t = np.array(img_data['translation']).reshape(3, 1)
        # Projecting 3d points onto the image plane
        projected_points = intrinsics @ (rot_matrix @ np.array(points3D)[:, 1:].T + t)
        projected_points = (projected_points[:2] / projected_points[2]).T.astype(int)
        # Checking if points are on the image plane
        valid_points = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < img_shape[1]) & \
                       (projected_points[:, 1] >= 0) & (projected_points[:, 1] < img_shape[0])
        # Draw reprojected points on the image plane
        for point in projected_points[valid_points]:
            cv2.circle(img, tuple(point), 4, (0, 0, 255), -1)
        
        cv2.imwrite(str(output / img_data['name']), img)

def main(images_folder, cameras_txt, images_txt, points3D_txt, camera_id, output):
    # Make output folder if not exists
    output.mkdir(parents=True, exist_ok=True)

    # Read camera parameters
    camera_params = read_cameras_txt(cameras_txt, camera_id)
    images_data = read_images_txt(images_txt)
    points3D = read_points3D_txt(points3D_txt)

    reproject_3d_points(images_folder, images_data, points3D, camera_params, camera_id, output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default= Path("preprocessed/undistort_right"), help = "Folder directory of the input images")
    parser.add_argument("--colmap_output", type=Path, default= Path("colmap_data/right"), help= "Folder directory of the colmap output, where the .bin files are stored")
    parser.add_argument("--camera_id", type = int, default = 4, help = "The ID of camera which can be found in the cameras.txt file")
    parser.add_argument("--output", type=Path, default= Path("reprojection_right/"), help = "Output path directory")
    args = parser.parse_args()

    cameras = args.colmap_output / "cameras.txt"
    images = args.colmap_output / "images.txt"  
    points3D = args.colmap_output / "points3D.txt" 

    main(args.images, cameras, images, points3D, args.camera_id, args.output)
