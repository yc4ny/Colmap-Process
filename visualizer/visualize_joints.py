import pickle 
import open3d as o3d
import numpy as np 
import os 
import glob 
import time
import json 

def align_joints_to_camera(joints, camera_location):
    # Calculate the translation vector needed to align the first joint with the camera location
    translation_vector = camera_location - joints[0]

    # Create a new array to store the aligned joints
    aligned_joints = np.zeros_like(joints)

    # Translate each joint in the array
    for i, joint in enumerate(joints):
        aligned_joints[i] = joint + translation_vector

    return aligned_joints

def create_hand_geometry(joints, connections, color=[1, 0, 0]):
    hand_pcd = o3d.geometry.PointCloud()
    hand_pcd.points = o3d.utility.Vector3dVector(joints)
    hand_pcd.paint_uniform_color(color)
    hand_pcd.estimate_normals()

    hand_lines = o3d.geometry.LineSet()
    hand_lines.points = hand_pcd.points
    hand_lines.lines = o3d.utility.Vector2iVector(connections)
    hand_lines.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in range(len(connections))])  # Blue color for the lines

    return hand_pcd, hand_lines


def prepare_frame_geometry(data, connections, extrinsics, scale=5):
    left_joints = data['pred_output_list'][0]['left_hand']['pred_joints_smpl']
    right_joints = data['pred_output_list'][0]['right_hand']['pred_joints_smpl']

    base_name = os.path.splitext(os.path.basename(data['image_path']))[0]
    base_name = base_name.replace('_prediction_result', '')

    head_extrinsic_matrix = extrinsics['head'][base_name + '.jpg']
    left_extrinsic_matrix = extrinsics['left'][base_name.replace('head', 'left') + '.jpg']
    right_extrinsic_matrix = extrinsics['right'][base_name.replace('head', 'right') + '.jpg']

    left_joints = align_joints_to_camera(left_joints * scale, -left_extrinsic_matrix[:, :3].T @ left_extrinsic_matrix[:, 3])
    right_joints = align_joints_to_camera(right_joints * scale, -right_extrinsic_matrix[:, :3].T @ right_extrinsic_matrix[:, 3])

    left = create_hand_geometry(left_joints, connections, color=[1, 0, 0])
    right = create_hand_geometry(right_joints, connections, color=[1, 0, 0])

    return left, right


def visualize_3d_points(pkl_files, connections, ply_file_path, scale=10, extrinsics=None):
    # Load the PLY file
    colmap_pcd = o3d.io.read_point_cloud(ply_file_path)
    colmap_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Grey color for the points from the PLY file

    viewer = o3d.visualization.Visualizer()
    viewer.create_window(window_name='Scene', width=2400, height=1800)
    viewer.add_geometry(colmap_pcd)
    viewer.run()
    param = viewer.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters("output_extrinsic/viewpoint.json", param)

    # Iterate through all the .pkl files, load the joint data, and visualize the hand movements sequentially
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        # Prepare and visualize the scene for the current frame
        frame_geometry = prepare_frame_geometry(data, connections, extrinsics, scale)
        for hand_geom in frame_geometry:
            for geom in hand_geom:
                viewer.add_geometry(geom)
                
        param = o3d.io.read_pinhole_camera_parameters("output_extrinsic/viewpoint.json")
        ctr = viewer.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)

        viewer.poll_events()
        viewer.update_renderer()
        time.sleep(0.1)

        # Remove current frame geometries before adding new ones
        for hand_geom in frame_geometry:
            for geom in hand_geom:
                viewer.remove_geometry(geom)
    viewer.destroy_window()

def main():
    connections = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [0, 9], 
        [9, 10],
        [10, 11],
        [11, 12],
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20]
    ]

    ply_file_path = 'colmap_data/right/points.ply'
    # Get all the .pkl files in the joints_3d/head/ folder
    pkl_files = sorted(glob.glob('joints_3d/head/*.pkl'))

    # Load extrinsics using the corresponding key for head, left, and right
    with open('output_extrinsic/head_extrinsic.pkl', 'rb') as f:
        head_extrinsics = pickle.load(f)

    with open('output_extrinsic/left_extrinsic.pkl', 'rb') as f:
        left_extrinsics = pickle.load(f)

    with open('output_extrinsic/right_extrinsic.pkl', 'rb') as f:
        right_extrinsics = pickle.load(f)

    extrinsics = {'head': head_extrinsics, 'left': left_extrinsics, 'right': right_extrinsics}

    # Call visualize_3d_points() with the list of pkl_files and extrinsics dictionary
    visualize_3d_points(pkl_files, connections, ply_file_path, scale=5, extrinsics=extrinsics)

if __name__ == "__main__":
    main()
