import json
import numpy as np
import cv2

def ComputePointJacobian(X, R,C):
    """
    Compute the point Jacobian

    Parameters
    ----------
    X : ndarray of shape (3,)
        3D point
    p : ndarray of shape (7,)
        Camera pose made of camera center and quaternion

    Returns
    -------
    dfdX : ndarray of shape (2, 3)
        The point Jacobian
    """
    x = np.matmul(np.column_stack((R,C)), X)
    u = x[0]
    v = x[1]
    w = x[2]
    du_dc = R[0, :]
    dv_dc = R[1, :]
    dw_dc = R[2, :]

    dfdX = np.stack([
        (w * du_dc - u * dw_dc) / (w**2),
        (w * dv_dc - v * dw_dc) / (w**2)
    ], axis=0)

    return dfdX

def Triangulation_nl(point3d,I1,I2,R1, C1, R2, C2, x1, x2):
    p1 = np.matmul(I1,np.column_stack((R1,C1)))
    p2 = np.matmul(I2,np.column_stack((R2,C2)))
    lamb = 0.1
    n_iter = 100
    X_new = point3d.copy()

    for i in range(0,point3d.shape[0]):
        pt = point3d[i,:]
        if pt[0] == 0:
            continue
        for j in range(n_iter):
            pt = np.append(pt,1)
            proj1 = np.matmul(p1,pt)
            proj1 = proj1[:2] / proj1[2]
            proj2 = np.matmul(p2,pt)
            proj2 = proj2[:2] / proj2[2]

            dfdX1 = ComputePointJacobian(pt, R1,C1)
            dfdX2 = ComputePointJacobian(pt, R2,C2)

            H1 = dfdX1.T @ dfdX1 + lamb * np.eye(3)
            H2 = dfdX2.T @ dfdX2 + lamb * np.eye(3)
            J1 = dfdX1.T @ (x1[i,:] - proj1)
            J2 = dfdX2.T @ (x2[i,:] - proj2)
            if np.linalg.det(H1) == 0 or np.linalg.det(H2) == 0:
                continue
            delta_pt = np.linalg.inv(H1) @ J1 + np.linalg.inv(H2) @ J2
            pt = pt[:3]
            pt += delta_pt

        X_new[i,:] = pt
    return X_new

def LinearTriangulation(K1,K2, C1, R1, C2, R2, x1, x2):

    I = np.identity(3)
    C1 = np.reshape(C1, (3, 1))
    C2 = np.reshape(C2, (3, 1))

    P1 = np.matmul(K1, np.column_stack((R1,C1)))
    P2 = np.matmul(K2, np.column_stack((R2,C2)))

    p1T = P1[0,:].reshape(1,4)
    p2T = P1[1,:].reshape(1,4)
    p3T = P1[2,:].reshape(1,4)

    p_dash_1T = P2[0,:].reshape(1,4)
    p_dash_2T = P2[1,:].reshape(1,4)
    p_dash_3T = P2[2,:].reshape(1,4)

    all_X = []
    for i in range(x1.shape[0]):
        if np.isnan(x1[i][0]) == True or np.isnan(x2[i][0])==True:
            all_X.append(([0,0,0,0]))
            continue 

        x = x1[i,0]
        y = x1[i,1]
        x_dash = x2[i,0]
        y_dash = x2[i,1]


        A = []
        A.append((y * p3T) -  p2T)
        A.append(p1T -  (x * p3T))
        A.append((y_dash * p_dash_3T) -  p_dash_2T)
        A.append(p_dash_1T -  (x_dash * p_dash_3T))

        A = np.array(A).reshape(4,4)

        _, _, vt = np.linalg.svd(A)
        v = vt.T
        x = v[:,-1]
        all_X.append(x)
    return np.array(all_X)

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
    """
    Calculate the average reprojection error.

    Parameters
    ----------
    reprojected_points : ndarray of shape (n, 2)
        Reprojected 2D points
    mediapipe_joints : ndarray of shape (n, 2)
        Original keypoints detected by MediaPipe

    Returns
    -------
    float
        The average reprojection error
    """
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

def main(): 

    # Loading Intrinsic Parameters
    fx_l, fy_l, cx_l, cy_l,k1_l, k2_l, p1_l, p2_l= read_cameras_txt("colmap_data/left/cameras.txt")
    left_intrinsic = np.array([
        [fx_l, 0, cx_l],
        [0, fy_l, cy_l],
        [0, 0, 1]
    ])

    fx_r, fy_r, cx_r, cy_r,k1_r, k2_r, p1_r, p2_r= read_cameras_txt("colmap_data/right/cameras.txt")
    right_intrinsic = np.array([
        [fx_r, 0, cx_r],
        [0, fy_r, cy_r],
        [0, 0, 1]
    ])

    # print(intrinsic_left)
    # print(intrinsic_right)   

    # Loading Extrinsic Parameters
    left_image = read_images_txt("colmap_data/left/images.txt")
    right_image = read_images_txt("colmap_data/right/images.txt")
    for i in range(len(left_image)):
        if left_image[i]['name'] == "left_00106.jpg":
            left_quaternion = left_image[i]['quaternion']
            left_translation = left_image[i]['translation']
    for i in range(len(right_image)):
        if right_image[i]['name'] == "right_00106.jpg":
            right_quaternion = right_image[i]['quaternion']
            right_translation = right_image[i]['translation']
    left_rotation = quaternion_to_rotation_matrix(left_quaternion)
    right_rotation = quaternion_to_rotation_matrix(right_quaternion)
    left_extrinsic = np.column_stack((left_rotation, left_translation))
    right_extrinsic = np.column_stack((right_rotation, right_translation))

    # Calculate Projection Matrix
    left_projectionMatrix = np.matmul(left_intrinsic, left_extrinsic)
    right_projectionMatrix = np.matmul(right_intrinsic, right_extrinsic)

    # Load Detected 2D Keypoints
    with open("detect_hand/undistort_left/left_00106_joints.json", 'r') as file:
        left_joints = json.load(file)
        l_leftjoints = np.array(left_joints['left'][0])
        l_rightjoints = np.array(left_joints['right'][0])

    with open("detect_hand/undistort_right/right_00106_joints.json", 'r') as file:
        right_joints = json.load(file)
        r_leftjoints = np.array(right_joints['left'][0])
        r_rightjoints = np.array(right_joints['right'][0])

    # LinearTriangulation(K1,K2, C1, R1, C2, R2, x1, x2)
    left_points3D = LinearTriangulation(left_intrinsic, right_intrinsic, left_translation, left_rotation, right_translation, right_rotation, l_leftjoints, r_leftjoints)
    left_image = cv2.imread("preprocessed/undistort_left/left_00106.jpg")

    # Reproject 3D points back onto the original left image
    reprojected_points = reproject_points(left_points3D[:, :3], left_projectionMatrix)
    print(reprojected_points)
    print("Average Reprojection Error: " + str(average_reprojection_error(reprojected_points, l_leftjoints)))
    
    # # Triangulation_nl(point3d,I1,I2,R1, C1, R2, C2, x1, x2):
    # optimized_points = Triangulation_nl(reprojected_points, left_intrinsic, right_intrinsic, left_rotation, left_translation, right_rotation, right_translation, l_leftjoints,r_leftjoints)
    # reprojected_points = reproject_points(optimized_points[:, :3], left_projectionMatrix)
    # print(reprojected_points)
    # print("Average Reprojection Error: " + str(average_reprojection_error(reprojected_points, l_leftjoints)))

if __name__ == "__main__":
    main()
