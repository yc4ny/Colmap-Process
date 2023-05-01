import cv2
import numpy as np
import os
from tqdm import tqdm

def read_cameras_txt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    camera_info = lines[-1].strip().split()
    return [float(val) for val in camera_info[4:12]]

def read_images_txt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    image_data = []
    for i in range(4, len(lines), 2):
        image_info = lines[i].strip().split()
        img_id, qw, qx, qy, qz, tx, ty, tz, cam_id, name = image_info
        image_data.append({
            'id': int(img_id),
            'quaternion': [float(qw), float(qx), float(qy), float(qz)],
            'translation': [float(tx), float(ty), float(tz)],
            'camera_id': int(cam_id),
            'name': name
        })
    return image_data

def read_points3D_txt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    points3D = []
    for line in lines[3:]:
        data = line.strip().split()
        point_id, x, y, z = map(float, data[:4])
        points3D.append([point_id, x, y, z])
    return points3D

def quaternion_to_rotation_matrix(quaternion):
    qw, qx, qy, qz = quaternion
    rot_matrix = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return rot_matrix

def undistort_points(points, k1, k2, p1, p2, fx, fy, cx, cy):
    undistorted_points = []
    for point in points:
        x, y = (point - np.array([cx, cy])) / np.array([fx, fy])
        r2 = x**2 + y**2
        x_distorted = x * (1 + k1 * r2 + k2 * r2**2) + 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
        y_distorted = y * (1 + k1 * r2 + k2 * r2**2) + p1 * (r2 + 2 * y**2) + 2 * p2 * x * y
        undistorted_points.append([x_distorted * fx + cx, y_distorted * fy + cy])
    return np.array(undistorted_points)

def reproject_3d_points(images_folder, images_data, points3D, camera_params):
    fx, fy, cx, cy, k1, k2, p1, p2 = camera_params
    intrinsics = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    for img_data in tqdm(images_data, desc="Processing frames"):
        if img_data['camera_id'] != 2:
            continue
        img = cv2.imread(os.path.join(images_folder, img_data['name']))
        rot_matrix = quaternion_to_rotation_matrix(img_data['quaternion'])
        t = np.array(img_data['translation']).reshape(3, 1)
        # print(rot_matrix)
        # print(t)
        projected_points = []
        for point in points3D:
            _, x, y, z = point
            proj_point = intrinsics @ (rot_matrix @ np.array([x, y, z]).reshape(3, 1) + t)
            proj_point = proj_point[:2] / proj_point[2]
            projected_points.append(proj_point.ravel())
        
        for point in projected_points:
            x_proj, y_proj = int(point[0]), int(point[1])

            if 0 <= x_proj < img.shape[1] and 0 <= y_proj < img.shape[0]:
                cv2.circle(img, (x_proj, y_proj), 4, (0, 0, 255), -1)

        cv2.imwrite(os.path.join( "preprocessed/reproject_opencv_left", f"{img_data['name']}"), img)

def main(images_folder, cameras_txt, images_txt, points3D_txt):
    camera_params = 1929.11, 1929.11, 1920, 1080,0, 0, 0, 0
    images_data = read_images_txt(images_txt)
    points3D = read_points3D_txt(points3D_txt)

    reproject_3d_points(images_folder, images_data, points3D, camera_params)

if __name__ == "__main__":
    images_folder = "preprocessed/undistort_opencv_left"  # Change this to the folder containing the images
    cameras_txt = "colmap_data/left/cameras.txt" # Change this to the path of your cameras.txt file
    images_txt = "colmap_data/left/images.txt"  # Change this to the path of your images.txt file
    points3D_txt = "colmap_data/left/points3D.txt" # Change this to the path of your points3D.txt file

    main(images_folder, cameras_txt, images_txt, points3D_txt)
