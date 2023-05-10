import numpy as np
import cv2 
import pickle 
import torch 
import torch.optim as optim
import json 
import argparse
import os 

def reproject_joints_on_image(image, joints3D, intrinsic, extrinsic):
    # Ensure joints is a homogeneous coordinate
    if joints3D.shape[1] == 3:
        joints3D = np.concatenate([joints3D, np.ones((joints3D.shape[0], 1))], axis=1)
    
    # Project the 3D points using the camera parameters
    points2D = intrinsic @ extrinsic @ joints3D.T
    
    # Convert to inhomogeneous coordinates
    points2D = points2D[:2, :] / points2D[2, :]
    points2D = points2D.T
    
    # Get image dimensions
    height, width = image.shape[:2]

    # Draw each joint on the image
    for i in range(points2D.shape[0]):
        x, y = int(points2D[i, 0]), int(points2D[i, 1])
        
        # Check if point is within image bounds
        if 0 <= x < width and 0 <= y < height:
            image = cv2.circle(image, (x, y), radius=1, color=(0, 255, 0), thickness=-1)
        
    return image

# Function to align the 3D joints to the camera center
def align_joints_to_camera(joints, camera_location):
    # Calculate the translation vector needed to align the first joint with the camera location
    translation_vector = camera_location - joints[0]

    # Create a new array to store the aligned joints
    aligned_joints = np.zeros_like(joints)

    # Translate each joint in the array
    for i, joint in enumerate(joints):
        aligned_joints[i] = joint + translation_vector

    return aligned_joints

# Define the projection function
def project_3d_to_2d(points_3d, intrinsic, extrinsic):
    ones = torch.ones(points_3d.shape[0], 1, dtype=torch.float32)
    points_3d_h = torch.cat([points_3d, ones], dim=1)
    projected = intrinsic @ (extrinsic @ points_3d_h.T)
    projected /= projected[2, :]
    return projected[:2, :].T

def reprojection_loss(left_extrinsic, head_extrinsic, left_intrinsic, head_intrinsic, left_2d, head_2d, joints3d):
    # convert all inputs to the same data type (float)
    left_extrinsic = left_extrinsic.float()
    head_extrinsic = head_extrinsic.float()
    left_intrinsic = left_intrinsic.float()
    head_intrinsic = head_intrinsic.float()
    left_2d = left_2d.float()
    head_2d = head_2d.float()
    joints3d = joints3d.float()

    # ensure that joints3d has an extra dimension of size 1 at the end
    if len(joints3d.shape) == 2:
        joints3d = torch.unsqueeze(joints3d, -1)

    # create homogeneous coordinates for 3d joints
    ones = torch.ones(joints3d.shape[0], 1, 1).to(joints3d.device)
    joints3d_homo = torch.cat([joints3d, ones], dim=1)

    # squeeze the last dimension
    joints3d_homo = torch.squeeze(joints3d_homo, -1)

    # calculate predicted 2d points for each camera view
    left_pred_homo = torch.matmul(left_extrinsic, joints3d_homo.T).T
    head_pred_homo = torch.matmul(head_extrinsic, joints3d_homo.T).T

    left_pred = torch.matmul(left_intrinsic, left_pred_homo[:,:3].T).T
    head_pred = torch.matmul(head_intrinsic, head_pred_homo[:,:3].T).T

    # convert from homogeneous to cartesian coordinates
    left_pred = left_pred[:, :2] / left_pred[:, 2:3]
    head_pred = head_pred[:, :2] / head_pred[:, 2:3]

    # calculate mean squared error between predicted and actual 2d points
    left_loss = torch.mean((left_pred - left_2d)**2)
    head_loss = torch.mean((head_pred - head_2d)**2)

    # return the sum of the losses for each camera view
    return left_loss + head_loss





def main():

    # Argument parser
    parser = argparse.ArgumentParser(description="Reproject 3D joints on 2D images.")
    parser.add_argument('--frame', type=str, required=False, help="Frame number. E.g., '111'", default = "1")
    args = parser.parse_args()
    
    frame_str = str(args.frame).zfill(5)

    head_intrinsic = np.array([
        [481.96,    0,      480],
        [0,     481.96,     270],
        [0,         0,        1]
    ])

    left_intrinsic = np.array([
        [467.488,    0,      480],
        [0,     467.488,     270],
        [0,         0,        1]
    ])

    right_intrinsic = np.array([
        [ 483.619,    0,      480],
        [0,      483.619,     270],
        [0,         0,        1]
    ])
    with open(f'data/fridge/camera_extrinsic/left_extrinsic.pkl', 'rb') as f:
       left_extrinsic = pickle.load(f)[f'left_{frame_str}.jpg']
    with open(f'data/fridge/camera_extrinsic/right_extrinsic.pkl', 'rb') as f:
       right_extrinsic = pickle.load(f)[f'right_{frame_str}.jpg']
    with open(f'data/fridge/camera_extrinsic/head_extrinsic.pkl', 'rb') as f:
       head_extrinsic = pickle.load(f)[f'head_{frame_str}.jpg']

    # Load 3D joints for left and right hand
    with open(f'data/fridge/frankmocap_joints/head_{frame_str}_prediction_result.pkl', 'rb') as f:
        data = pickle.load(f)
        left_joints3d = data['pred_output_list'][0]['left_hand']['pred_joints_smpl']
        right_joints3d = data['pred_output_list'][0]['right_hand']['pred_joints_smpl']

    # Initial Joint 
    left_joints3d = align_joints_to_camera(left_joints3d, left_extrinsic[:, :3].T @ left_extrinsic[:, 3])
    right_joints3d = align_joints_to_camera(right_joints3d, right_extrinsic[:, :3].T @ right_extrinsic[:, 3])

    # Load 2D annotations for left and right hand
    with open(f"left_output/left_{frame_str}_joints.json", 'r') as f:
        left_2d = json.load(f)['left'][0]
    with open(f"right_output/right_{frame_str}_joints.json", 'r') as f:
        right_2d = json.load(f)['right'][0]
    with open(f"head_output/head_{frame_str}_joints.json", 'r') as f:
        head_left_2d = json.load(f)['left'][0]
    with open(f"head_output/head_{frame_str}_joints.json", 'r') as f:
        head_right_2d = json.load(f)['right'][0]

    left_intrinsic = torch.from_numpy(left_intrinsic)
    right_intrinsic = torch.from_numpy(right_intrinsic)
    head_intrinsic = torch.from_numpy(head_intrinsic)
    left_extrinsic = torch.from_numpy(left_extrinsic)
    right_extrinsic = torch.from_numpy(right_extrinsic)
    head_extrinsic = torch.from_numpy(head_extrinsic)
    left_joints3d = torch.from_numpy(left_joints3d)
    right_joints3d = torch.from_numpy(right_joints3d)
    left_2d = torch.tensor(left_2d)
    right_2d = torch.tensor(right_2d)
    head_left_2d = torch.tensor(head_left_2d)
    head_right_2d = torch.tensor(head_right_2d)

    # Optimized parameters 
    scale = torch.tensor([1], dtype = torch.float32, requires_grad=True) 
    translation = torch.zeros([1,3], dtype=torch.float32, requires_grad=True)
    # Set optimizer
    optimizer = optim.Adam([scale, translation], lr=0.01)

    # Optimize the 3D joints iteratively to minimize the loss
    num_iterations = 10000
    for i in range(num_iterations):
        optimizer.zero_grad()
        
        # Apply scale and translation to joints3d before computing the loss
        transformed_left_joints3d = scale * left_joints3d + translation
        
        loss = reprojection_loss(left_extrinsic, head_extrinsic, left_intrinsic, head_intrinsic, left_2d, head_left_2d, transformed_left_joints3d)
        loss.backward()
        optimizer.step()
        
        if i % 1 == 0:
            print(f'Iteration {i}, Loss: {loss.item()}')
    
    left_joints_3d = transformed_left_joints3d.detach().numpy()
    print("Left scale: " +  str(scale))
    print("Left translation: " + str(translation))   

    # Optimized parameters 
    scale = torch.tensor([1], dtype = torch.float32, requires_grad=True) 
    translation = torch.zeros([1,3], dtype=torch.float32, requires_grad=True)
    # Set optimizer
    optimizer = optim.Adam([scale, translation], lr=0.01)
    # Optimize the 3D joints iteratively to minimize the loss
    num_iterations = 10000
    for i in range(num_iterations):
        optimizer.zero_grad()

        # Apply scale and translation to joints3d before computing the loss
        transformed_right_joints3d = scale * right_joints3d + translation

        loss = reprojection_loss(right_extrinsic, head_extrinsic, right_intrinsic, head_intrinsic, right_2d, head_right_2d, transformed_right_joints3d)
        loss.backward()
        optimizer.step()
        
        if i % 1 == 0:
            print(f'Iteration {i}, Loss: {loss.item()}')

    right_joints_3d = transformed_right_joints3d.detach().numpy()
    print("Right scale: " +  str(scale))
    print("Right translation: " + str(translation)) 
    
    data = {
        'left': left_joints_3d,
        'right': right_joints_3d
    }
    print(data)
    with open('output_00111.pkl', 'wb') as f:
        pickle.dump(data, f)

    # Check reprojection on image
    img_left = cv2.imread(f"data/fridge/img/undistort_left/left_{frame_str}.jpg")
    output_left = reproject_joints_on_image(img_left, left_joints_3d, left_intrinsic, left_extrinsic)
    cv2.imwrite("output_left.jpg", output_left)
    img_head = cv2.imread(f"data/fridge/img/undistort_head/head_{frame_str}.jpg")
    output_head = reproject_joints_on_image(img_head, left_joints_3d, head_intrinsic, head_extrinsic)
    output_head = reproject_joints_on_image(img_head, right_joints_3d, head_intrinsic, head_extrinsic)
    cv2.imwrite("output_head.jpg", output_head)
    img_right = cv2.imread(f"data/fridge/img/undistort_right/right_{frame_str}.jpg")
    output_right = reproject_joints_on_image(img_right, right_joints_3d, right_intrinsic, right_extrinsic)
    cv2.imwrite("output_right.jpg", output_right)


if __name__ == "__main__":
    main()