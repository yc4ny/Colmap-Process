import json
import numpy as np
import cv2

def triangulate_points(json_file1, json_file2, extrinsics1, translation1, intrinsics1, extrinsics2, translation2, intrinsics2):
    def undistort_points(points, k1, k2, p1, p2, fx, fy, cx, cy):
        undistorted_points = []
        for point in points:
            x, y = (point - np.array([cx, cy])) / np.array([fx, fy])
            r2 = x**2 + y**2
            x_distorted = x * (1 + k1 * r2 + k2 * r2**2) + 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
            y_distorted = y * (1 + k1 * r2 + k2 * r2**2) + p1 * (r2 + 2 * y**2) + 2 * p2 * x * y
            undistorted_points.append([x_distorted * fx + cx, y_distorted * fy + cy])
        return np.array(undistorted_points)

    with open(json_file1) as f:
        joints1 = json.load(f)
    with open(json_file2) as f:
        joints2 = json.load(f)

    # Convert intrinsic parameters to camera matrix and distortion coefficients
    fx1, fy1, cx1, cy1, k1_1, k2_1, p1_1, p2_1 = intrinsics1
    fx2, fy2, cx2, cy2, k1_2, k2_2, p1_2, p2_2 = intrinsics2

    # Build the projection matrices for both cameras
    projection_matrix1 = np.hstack((extrinsics1, translation1))
    projection_matrix2 = np.hstack((extrinsics2, translation2))

    camera_matrix1 = np.array([[fx1, 0, cx1], [0, fy1, cy1], [0, 0, 1]])
    camera_matrix2 = np.array([[fx2, 0, cx2], [0, fy2, cy2], [0, 0, 1]])

    projection_matrix1 = camera_matrix1 @ projection_matrix1
    projection_matrix2 = camera_matrix2 @ projection_matrix2

    joints_3d = {"left": [], "right": []}

    for side in ["left", "right"]:
        for i in range(len(joints1[side][0])): # Modify this line to access the inner list
            points1 = np.array(joints1[side][0][i]).T
            points2 = np.array(joints2[side][0][i]).T

            points1_undistorted = undistort_points(points1, k1_1, k2_1, p1_1, p2_1, fx1, fy1, cx1, cy1)
            points2_undistorted = undistort_points(points2, k1_2, k2_2, p1_2, p2_2, fx2, fy2, cx2, cy2)

            points_3d_homogeneous = cv2.triangulatePoints(projection_matrix1, projection_matrix2, points1_undistorted.T, points2_undistorted.T)
            points_3d = cv2.convertPointsFromHomogeneous(points_3d_homogeneous.T).squeeze()
            joints_3d[side].append(points_3d.tolist())

    return joints_3d


if __name__ == "__main__":
    json_file1 = "detect_hand/left/left_00119_joints.json"
    json_file2 = "detect_hand/right/right_00119_joints.json"

    extrinsics1 = np.array([[0.8520928, 0.31747912, -0.41610712],
                            [0.00859511, 0.78642624, 0.61762471],
                            [0.52332041, -0.52985014, 0.66737935]])
    translation1 = np.array([[1.8237], [-2.73237], [0.138372]])
    intrinsics1 = [1745.52, 1744.58, 1920, 1080, -0.183042, 0.0248962, -0.00013907, 0.000305203]

    extrinsics2 = np.array([[0.93940081, 0.06059739, 0.33742267],
                            [-0.30099994, 0.61689943, 0.72720974],
                            [-0.16408881, -0.78470563, 0.59775564]])
    translation2 = np.array([[-2.51374], [-2.12002], [0.973812]])
    intrinsics2 = [1760.56, 1742.97, 1920, 1080, -0.187055, 0.0260906, -0.00198784, -0.00131332]

    joints_3d = triangulate_points(json_file1, json_file2, extrinsics1, translation1, intrinsics1, extrinsics2, translation2, intrinsics2)

    output_path = "output_3d_joints.json"
    with open(output_path, "w") as outfile:
        json.dump(joints_3d, outfile)