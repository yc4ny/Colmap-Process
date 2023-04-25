import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def quaternion_to_rotation_matrix(quaternion):
    qw, qx, qy, qz = quaternion
    rot_matrix = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return rot_matrix

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

def reproject_points(points3D, projection_matrix):
    points3D_homogeneous = np.column_stack((points3D, np.ones(points3D.shape[0])))
    projected_points = np.matmul(projection_matrix, points3D_homogeneous.T).T
    projected_points = projected_points[:, :2] / projected_points[:, 2, np.newaxis]
    return projected_points


def average_reprojection_error(reprojected_points, mediapipe_joints):
    assert reprojected_points.shape == mediapipe_joints.shape, "Both input arrays must have the same shape"
    
    num_points = reprojected_points.shape[0]
    errors = np.sqrt(np.sum((reprojected_points - mediapipe_joints) ** 2, axis=1))
    average_error = np.mean(errors)
    
    return average_error

def draw_points(image, points, color=(0, 255, 0), radius=5):
    for point in points:
        cv2.circle(image, (int(point[0]), int(point[1])), radius, color, -1)

def undistort_points(points, intrinsic, distortion_coeffs):
    return cv2.undistortPoints(points.reshape(-1, 1, 2).astype(np.float32), intrinsic, distortion_coeffs).reshape(-1, 2)

def distort_points(points, intrinsic, distortion_coeffs):
    distorted_points = cv2.projectPoints(points.reshape(-1, 1, 3).astype(np.float32), np.zeros(3), np.zeros(3), intrinsic, distortion_coeffs)[0]
    return distorted_points.reshape(-1, 2)


def visualize_3d_joints(joints_3D):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for point in joints_3D:
        if not (np.any(np.isnan(point)) or np.any(np.isinf(point))):
            ax.scatter(point[0], point[1], point[2], c='r', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def main(): 

    # # Loading Intrinsic Parameters
    # fx_l, fy_l, cx_l, cy_l,k1_l, k2_l, p1_l, p2_l= read_cameras_txt("colmap_data/left/cameras.txt")
    left_intrinsic = np.array([
        [1803.52, 0, 1920],
        [0, 1803.52, 1080],
        [0, 0, 1]
    ])

    # fx_r, fy_r, cx_r, cy_r,k1_r, k2_r, p1_r, p2_r= read_cameras_txt("colmap_data/right/cameras.txt")
    right_intrinsic = np.array([
        [1733.9, 0, 1920],
        [0, 1733.9, 1080],
        [0, 0, 1]
    ])

    # print(intrinsic_left)
    # print(intrinsic_right)   

    # Loading Extrinsic Parameters
    left_image = read_images_txt("colmap_data/right/images.txt")
    right_image = read_images_txt("colmap_data/right/images.txt")
    for i in range(len(left_image)):
        if left_image[i]['name'] == "left_00165.jpg":
            left_quaternion = left_image[i]['quaternion']
            left_translation = left_image[i]['translation']
    for i in range(len(right_image)):
        if right_image[i]['name'] == "right_00165.jpg":
            right_quaternion = right_image[i]['quaternion']
            right_translation = right_image[i]['translation']
    left_rotation = quaternion_to_rotation_matrix(left_quaternion)
    right_rotation = quaternion_to_rotation_matrix(right_quaternion)
    left_extrinsic = np.column_stack((left_rotation, left_translation))
    right_extrinsic = np.column_stack((right_rotation, right_translation))

    # Calculate Projection Matrix
    left_projectionMatrix = np.matmul(left_intrinsic, left_extrinsic)
    right_projectionMatrix = np.matmul(right_intrinsic, right_extrinsic)

    # l_leftjoints = np.array([
    #     [2685,549],
    #     [3004,947],
    #     [2831,1064],
    #     [2747,1159],
    #     [2541,1355],
    #     [2528,1185],
    #     [2306,1508],
    #     [2345,1149],
    #     [2159,1544],
    #     [2150,1058],
    #     [1889,1472]
    # ])
    # r_leftjoints = np.array([
    #     [141,6],
    #     [857,184],
    #     [1040,389],
    #     [607,323],
    #     [1148,678],
    #     [493,317],
    #     [1051,764],
    #     [516,373],
    #     [915,725],
    #     [502,339],
    #     [671,556]
    # ])
    # Load Detected 2D Keypoints

    with open("detect_hand/left/left_00443_joints.json", 'r') as file:
        left_joints = json.load(file)
        l_leftjoints = np.array(left_joints['left'][0])
        # l_rightjoints = np.array(left_joints['right'][0])

    with open("detect_hand/right/right_00443_joints.json", 'r') as file:
        right_joints = json.load(file)
        r_leftjoints = np.array(right_joints['left'][0])
        # r_rightjoints = np.array(right_joints['right'][0])

    # Triangulate the 2D joints to obtain the 3D joints
    leftjoints_3D = cv2.triangulatePoints(left_projectionMatrix, right_projectionMatrix, l_leftjoints.T, r_leftjoints.T)

    # Convert the 3D joints from homogeneous to Euclidean coordinates
    leftjoints_3D = leftjoints_3D[:3, :] / leftjoints_3D[3, :]

    print("3D Joints:\n", leftjoints_3D.T)
    leftjoints_3D_list = leftjoints_3D.T.tolist()
    output_dict = {'3D_Joints': leftjoints_3D_list}

    # Write the dictionary to a JSON file
    with open('output.json', 'w') as outfile:
        json.dump(output_dict, outfile, indent=4)
    
    projected_points = reproject_points(leftjoints_3D.T, left_projectionMatrix)

if __name__ == "__main__":
    main()
