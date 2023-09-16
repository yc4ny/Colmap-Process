import pickle 
import numpy as np 
import cv2 

def main():
    
    intrinsics_head = np.array([
        [1966.57, 0, 1920],
        [0, 1966.57, 1080],
        [0, 0, 1]
    ])
    intrinsics_left = np.array([
        [2046.3, 0, 1920],
        [0, 2046.3, 1080],
        [0, 0, 1]
    ]) 
    intrinsics_right = np.array([
        [1772.88, 0, 1920],
        [0, 1772.88, 1080],
        [0, 0, 1]
    ])
    connections = [
        [0, 1],[1, 2],[2, 3],[3, 4],[0, 5],[5, 6],[6, 7],[7, 8],[0, 9],[9, 10],[10, 11],[11, 12],
        [0, 13],[13, 14],[14, 15],[15, 16],[0, 17],[17, 18],[18, 19],[19, 20]]
    
    with open("data/new/camera_extrinsic/head_extrinsic.pkl", 'rb') as f:
        head_extrinsic = pickle.load(f)   
    with open("data/new/camera_extrinsic/left_extrinsic.pkl", 'rb') as f:
        left_extrinsic = pickle.load(f)    
    with open("data/new/camera_extrinsic/right_extrinsic.pkl", 'rb') as f:
        right_extrinsic = pickle.load(f)  

    with open("initial_left_joints.pkl", 'rb') as f:
        left_joints = pickle.load(f)   
    with open("initial_right_joints.pkl", 'rb') as f:
        right_joints = pickle.load(f)

    img = cv2.imread('preprocessed/undistort_head/head_3_00001.jpg')  # load your image here

    joint_dict = {}  # This dict will store the 2D joint positions

    for left_joint in left_joints:
        left_3d = left_joints[left_joint]
        right_3d = right_joints[left_joint.replace("left", "right")]
        head_rot_matrix =  head_extrinsic[left_joint.replace("left", "head")][:, 0:3]
        head_translation = head_extrinsic[left_joint.replace("left", "head")][: , 3]
        head_projected_left = (intrinsics_head @ ((head_rot_matrix @ left_3d.T).T + head_translation).T)
        head_projected_left = head_projected_left[:2] / head_projected_left[2]
        
        joint_dict[left_joint] = head_projected_left.astype(int)  # store joint position to dictionary

        # cv2.circle(img, tuple(head_projected_left.astype(int)), 5, (0, 255, 0), -1)  # draw the joint

    for connection in connections:
        start_joint = joint_dict['left_joint' + str(connection[0])]
        end_joint = joint_dict['left_joint' + str(connection[1])]
        cv2.line(img, tuple(start_joint), tuple(end_joint), (0, 0, 255), 2)  # draw the connection

    cv2.imshow('Projected Joints', img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()