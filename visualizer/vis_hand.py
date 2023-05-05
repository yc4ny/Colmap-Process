import pickle 
import open3d as o3d
import numpy as np 
import os 
import glob 
import time
import json 
import argparse 

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
def prepare_frame_geometry(data, connections, extrinsics, scale=5):
    frame_geometry = []
    base_name = os.path.splitext(os.path.basename(data['image_path']))[0]
    base_name = base_name.replace('_prediction_result', '')

    head_key = base_name + '.jpg'
    left_key = base_name.replace('head', 'left') + '.jpg'
    right_key = base_name.replace('head', 'right') + '.jpg'

    if head_key not in extrinsics['head'] or left_key not in extrinsics['left'] or right_key not in extrinsics['right']:
        return frame_geometry

    if 'pred_joints_smpl' in data['pred_output_list'][0]['left_hand']:
        left_joints = data['pred_output_list'][0]['left_hand']['pred_joints_smpl']

        head_extrinsic_matrix = extrinsics['head'][head_key]
        left_extrinsic_matrix = extrinsics['left'][left_key]

        left_joints = align_joints_to_camera(left_joints * scale, -left_extrinsic_matrix[:, :3].T @ left_extrinsic_matrix[:, 3])

        left = create_hand_geometry(left_joints, connections, color=[1, 0, 0])
        frame_geometry.append(left)

    if 'pred_joints_smpl' in data['pred_output_list'][0]['right_hand']:
        right_joints = data['pred_output_list'][0]['right_hand']['pred_joints_smpl']

        head_extrinsic_matrix = extrinsics['head'][head_key]
        right_extrinsic_matrix = extrinsics['right'][right_key]

        right_joints = align_joints_to_camera(right_joints * scale, -right_extrinsic_matrix[:, :3].T @ right_extrinsic_matrix[:, 3])

        right = create_hand_geometry(right_joints, connections, color=[1, 0, 0])
        frame_geometry.append(right)

    return frame_geometry

def load_view(json_path): 
    with open(json_path, 'r') as f:
        view_params = json.load(f)
    trajectory = view_params["trajectory"][0]
    return trajectory['field_of_view'], trajectory['front'], trajectory['lookat'], trajectory['up'], trajectory['zoom']


def visualize_3d_points(pkl_files, connections, ply_file_path, scale=10, extrinsics=None, capture = None, output = None):
    # Load the PLY file
    colmap_pcd = o3d.io.read_point_cloud(ply_file_path)
    colmap_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Grey color for the points from the PLY file

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Scene', width=2400, height=1800)

    vis.add_geometry(colmap_pcd)
    field_of_view, front, lookat, up, zoom = load_view(f"data/views/view_{capture}.json")
    ctr = vis.get_view_control()

    # Initialize a counter for saving frames of visualized hand joints
    counter = 0
    # Iterate through all the .pkl files, load the joint data, and visualize the hand movements sequentially
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        # Prepare and visualize the scene for the current frame
        print(pkl_file)

        # Add all the necessary geometries in the frame
        frame_geometry = prepare_frame_geometry(data, connections, extrinsics, scale)
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
        time.sleep(1/30)

        # Save the image for current frame
        vis.capture_screen_image(f"visualizer/vis_img/{counter:05}.jpg")
        
        # Increment the counter
        counter += 1

        # Remove current geometries before adding new ones
        for hand_geom in frame_geometry:
            for geom in hand_geom:
                vis.remove_geometry(geom)

    vis.destroy_window()

def main():

    parser = argparse.ArgumentParser(description='Visualize 3D Hand in 3D Space.')
    parser.add_argument('--capture', type=str, required=True, help='Name of the captured data.', default = "desk")
    parser.add_argument('--output', type=str, required=True, help='Save directory.', default = "visualizer/vis_img")
    args = parser.parse_args()

    # Hand joint connections
    # https://github.com/facebookresearch/frankmocap/blob/main/docs/joint_order.md
    connections = [
        [0, 1],[1, 2],[2, 3],[3, 4],[0, 5],[5, 6],[6, 7],[7, 8],[0, 9],[9, 10],[10, 11],[11, 12],
        [0, 13],[13, 14],[14, 15],[15, 16],[0, 17],[17, 18],[18, 19],[19, 20]]

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

    # Run the Open3D visualizer for visualizing 3D hand in the reconstructed 3D scene from colmap
    visualize_3d_points(pkl_files, connections, ply_file_path, scale=5, extrinsics=extrinsics, capture = args.capture, output = args.output)

if __name__ == "__main__":
    main()