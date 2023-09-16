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

import pickle 
import open3d as o3d
import numpy as np 
import os 
import glob 
import time
import json 
import argparse 
import cv2 

def plot_camera_position(intrinsic, extrinsic, img_width, img_height, depth=0.4, color=[1, 0, 0]):
    # Extract rotation matrix and translation vector
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]

    # Compute camera center in world coordinates
    camera_center = -np.dot(R.T, t)

    # Compute the corners of the image plane in camera coordinates
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    corners_camera = depth * np.array([
        [(0 - cx) / fx, (0 - cy) / fy, 1],  # Top-left corner
        [(0 - cx) / fx, (img_height - 1 - cy) / fy, 1],  # Bottom-left corner
        [(img_width - 1 - cx) / fx, (img_height - 1 - cy) / fy, 1],  # Bottom-right corner
        [(img_width - 1 - cx) / fx, (0 - cy) / fy, 1],  # Top-right corner
    ])

    # Transform the corners to world coordinates
    corners_world = np.dot(R.T, corners_camera.T - t.reshape(-1, 1)).T

    # Create lines between the camera center and the corners, and between the corners to form the frustum
    points = [camera_center] + list(corners_world)
    lines = [[0, i] for i in range(1, 5)] + [[i, i % 4 + 1] for i in range(1, 5)]

    # Create a line set
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )

    # Color the lines
    line_set.paint_uniform_color(color)

    return line_set

def visualize_3d_points(pkl_files, connections, ply_file_path, scale=10,intrinsic = None, extrinsics=None, capture = None, output = None, width = None, height = None, fps = 30 ):

    head_extrin = extrinsics['head']
    left_extrin = extrinsics['left']
    right_extrin = extrinsics['right']
    # Load the camera parameters for the 'head', 'left' and 'right' cameras
    intrin_head = intrinsic['head']['intrinsic']
    intrin_left = intrinsic['left']['intrinsic']
    intrin_right = intrinsic['right']['intrinsic']

    # For saving initial joints (scale translation initial)
    all_left_joints = {}
    all_right_joints = {}

    # Load the PLY file``
    colmap_pcd = o3d.io.read_point_cloud(ply_file_path)
    colmap_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Grey color for the points from the PLY file

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    render_option = vis.get_render_option()
    render_option.point_size = 0.01  # Set the point size

    vis.add_geometry(colmap_pcd)
    field_of_view, front, lookat, up, zoom = load_view(f"data/views/view_{capture}.json")
    ctr = vis.get_view_control()

    # Initialize a counter for saving frames of visualized hand joints
    counter = 0
    # Initialize the first lines geometry
    lines_head = lines_left = lines_right = None

    # Iterate through all the .pkl files, load the joint data, and visualize the hand movements sequentially
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        # Prepare and visualize the scene for the current frame
        base_camera_name = os.path.basename(pkl_file)
        base_camera_name = "_".join(base_camera_name.split("_")[:2])
        base_camera_name = base_camera_name + '.jpg'
        left_filename = base_camera_name.replace('head', 'left')
        right_filename = base_camera_name.replace('head', 'right')

        if base_camera_name in head_extrin:
            new_lines_head = plot_camera_position(intrin_head, head_extrin[base_camera_name], img_width=960, img_height=540)
            if lines_head is not None:
                vis.remove_geometry(lines_head)
            vis.add_geometry(new_lines_head)
            lines_head = new_lines_head

            # If there is a corresponding 'left' frame, update the 'left' lines
            if left_filename in left_extrin:
                new_lines_left = plot_camera_position(intrin_left, left_extrin[left_filename], img_width=960, img_height=540)
                if lines_left is not None:
                    vis.remove_geometry(lines_left)
                vis.add_geometry(new_lines_left)
                lines_left = new_lines_left

            # If there is a corresponding 'right' frame, update the 'right' lines
            if right_filename in right_extrin:
                new_lines_right = plot_camera_position(intrin_right, right_extrin[right_filename], img_width=960, img_height=540)
                if lines_right is not None:
                    vis.remove_geometry(lines_right)
                vis.add_geometry(new_lines_right)
                lines_right = new_lines_right

        # Add all the necessary geometries in the frame
        frame_geometry, left_joints, right_joints = prepare_frame_geometry(data, connections, extrinsics, scale)

        # Extract base name from the full path
        base_path = os.path.basename(pkl_file)
        # Remove extension and replace it with .jpg
        base_name, _ = os.path.splitext(base_path)
        base_name = base_name.replace("_prediction_result", "")
        jpg_file = base_name + '.jpg'

        # Add current frame's joints to the dictionaries
        if left_joints is not None:
            all_left_joints[jpg_file] = left_joints
        if right_joints is not None:
            all_right_joints[jpg_file] = right_joints

        for hand_geom in frame_geometry:
            for geom in hand_geom:
                vis.add_geometry(geom)
        
        # Change field of view obtained from get_view function 
        ctr.change_field_of_view(field_of_view)
        ctr.set_front(front)
        ctr.set_lookat(lookat)
        ctr.set_up(up)
        ctr.set_zoom(zoom)

        # Update Renderer
        vis.poll_events()
        vis.update_renderer()

        # Set framerate: 1/30 = 30fps
        time.sleep(1/fps)

        # Save the image for current frame
        vis.capture_screen_image(f"{output}/{counter:05}.jpg")
        
        # Increment the counter
        counter += 1

        # Remove current geometries before adding new ones
        for hand_geom in frame_geometry:
            for geom in hand_geom:
                vis.remove_geometry(geom)

    # After the loop, save all joints to their respective files
    with open('initial_left_joints.pkl', 'wb') as f:
        pickle.dump(all_left_joints, f)
    with open('initial_right_joints.pkl', 'wb') as f:
        pickle.dump(all_right_joints, f)

    vis.destroy_window()

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

# Function to create joints and connections in 3D space 
def create_hand_geometry(joints, connections, color=[1, 0, 0]):
    hand_pcd = o3d.geometry.PointCloud()
    hand_pcd.points = o3d.utility.Vector3dVector(joints)
    hand_pcd.paint_uniform_color(color) # Red color for joint locations
    hand_pcd.estimate_normals()

    hand_lines = o3d.geometry.LineSet()
    hand_lines.points = hand_pcd.points
    hand_lines.lines = o3d.utility.Vector2iVector(connections)
    hand_lines.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in range(len(connections))])  # Blue color for the lines/connections

    return hand_pcd, hand_lines

# Function to extract camera extrinsics, 3D joints from input files and align, create geometries of the final geometries which needs to be rendered
def prepare_frame_geometry(data, connections, extrinsics, scale):
    frame_geometry = []
    base_name = os.path.splitext(os.path.basename(data['image_path']))[0]
    base_name = base_name.replace('_prediction_result', '')

    head_key = base_name + '.jpg'
    left_key = base_name.replace('head', 'left') + '.jpg'
    right_key = base_name.replace('head', 'right') + '.jpg'
    left_joints = None 
    right_joints = None

    if head_key not in extrinsics['head'] or left_key not in extrinsics['left'] or right_key not in extrinsics['right']:
        return frame_geometry, left_joints, right_joints

    if 'pred_joints_smpl' in data['pred_output_list'][0]['left_hand']:
        left_joints = data['pred_output_list'][0]['left_hand']['pred_joints_smpl']

        head_extrinsic_matrix = extrinsics['head'][head_key]
        left_extrinsic_matrix = extrinsics['left'][left_key]
        left_joints = align_joints_to_camera(left_joints * scale, -left_extrinsic_matrix[:, :3].T @ left_extrinsic_matrix[:, 3])
        left_joints = left_joints + np.array([2.8923, -8.8810, -7.0341]) * -0.1
        left = create_hand_geometry(left_joints, connections, color=[1, 0, 0])
        frame_geometry.append(left)

    if 'pred_joints_smpl' in data['pred_output_list'][0]['right_hand']:
        right_joints = data['pred_output_list'][0]['right_hand']['pred_joints_smpl']

        head_extrinsic_matrix = extrinsics['head'][head_key]
        right_extrinsic_matrix = extrinsics['right'][right_key]
        right_joints = right_joints + np.array([10.9484, -6.4760, -6.5716] )
        right_joints = align_joints_to_camera(right_joints * 3.9138, -right_extrinsic_matrix[:, :3].T @ right_extrinsic_matrix[:, 3])
        right_joints = right_joints + np.array([2.9484, -6.4760, -6.5716] ) * -0.1 
        # right_joints = align_joints_to_camera(right_joints * scale, -right_extrinsic_matrix[:, :3].T @ right_extrinsic_matrix[:, 3])
        right = create_hand_geometry(right_joints, connections, color=[1, 0, 0])
        frame_geometry.append(right)

    return frame_geometry, left_joints, right_joints

def load_view(json_path): 
    with open(json_path, 'r') as f:
        view_params = json.load(f)
    trajectory = view_params["trajectory"][0]
    return trajectory['field_of_view'], trajectory['front'], trajectory['lookat'], trajectory['up'], trajectory['zoom']


def get_camera_params(filename):
    # Define a dictionary to hold the camera parameters
    camera_params = {'head': None, 'left': None, 'right': None}

    # Open the file and read the lines
    with open(filename, 'r') as file:
        lines = file.readlines()
        
    # Process each line
    for line in lines:
        # Skip lines starting with '#'
        if line.startswith('#'):
            continue

        # Split the line into values
        values = line.split()

        # Extract the values
        camera_id = int(values[0])
        params = [float(value) for value in values[4:]]
        fx = params[0]
        cx = params[1]
        cy = params[2] 
        I = np.array([
            [fx, 0, cx],
            [0, fx, cy],
            [0, 0, 1]
        ])
        # Check the camera id and add the parameters to the dictionary
        if camera_id == 2:  # head
            camera_params['head'] = {'intrinsic' : I}
        elif camera_id == 3:  # left
            camera_params['left'] = {'intrinsic' : I}
        elif camera_id == 4:  # right
            camera_params['right'] = {'intrinsic' : I}

    return camera_params

def main():

    parser = argparse.ArgumentParser(description='Visualize 3D Hand in 3D Space.')
    parser.add_argument('--capture', type=str, required=False, help='Name of the captured data.', default = "fridge")
    parser.add_argument('--output', type=str, required=False, help='Save directory.', default = "output_camera_hand_fridge")
    parser.add_argument('--fps', type=int, required=False, help='Fps of visualizer' ,default = 30)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    # Hand joint connections
    # https://github.com/facebookresearch/frankmocap/blob/main/docs/joint_order.md
    connections = [
        [0, 1],[1, 2],[2, 3],[3, 4],[0, 5],[5, 6],[6, 7],[7, 8],[0, 9],[9, 10],[10, 11],[11, 12],
        [0, 13],[13, 14],[14, 15],[15, 16],[0, 17],[17, 18],[18, 19],[19, 20]]

    # img = cv2.imread(f"data/{args.capture}/img/undistort_opencv_head")

    # Load scene pointcloud
    ply_file_path = f'data/{args.capture}/colmap_data/sparse/0/points.ply'
    # Load 3d hand keypoint outputs from frankmocap's hand mocap detector
    pkl_files = sorted(glob.glob(f'data/{args.capture}/frankmocap_joints/*.pkl'))

    #Load the saved extrinsic parameters from colmap's register-image (EPNP) (Head, Left Wrist, Right Wrist Cameras)
    with open(f'data/{args.capture}/camera_extrinsic/head_extrinsic.pkl', 'rb') as f:
        head_extrinsics = pickle.load(f)
    with open(f'data/{args.capture}/camera_extrinsic/left_extrinsic.pkl', 'rb') as f:
        left_extrinsics = pickle.load(f)
    with open(f'data/{args.capture}/camera_extrinsic/right_extrinsic.pkl', 'rb') as f:
        right_extrinsics = pickle.load(f)
    extrinsics = {'head': head_extrinsics, 'left': left_extrinsics, 'right': right_extrinsics}

    intrinsic= get_camera_params('data/fridge/colmap_data/right/cameras.txt')

    # Run the Open3D visualizer for visualizing 3D hand in the reconstructed 3D scene from colmap
    visualize_3d_points(pkl_files, connections, ply_file_path, scale=5,intrinsic = intrinsic,  extrinsics=extrinsics, capture = args.capture, output = args.output, fps = args.fps)

if __name__ == "__main__":
    main()