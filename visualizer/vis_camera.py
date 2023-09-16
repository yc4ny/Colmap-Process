import numpy as np
import open3d as o3d
import pickle 
import json
import os
import subprocess 
from tqdm import tqdm


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


def plot_camera_position(intrinsic, extrinsic, img_width, img_height, depth=0.2, color=[1, 0, 0]):
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

def load_view(json_path): 
    with open(json_path, 'r') as f:
        view_params = json.load(f)
    trajectory = view_params["trajectory"][0]
    return trajectory['field_of_view'], trajectory['front'], trajectory['lookat'], trajectory['up'], trajectory['zoom']

if __name__ == "__main__":
    camera_params = get_camera_params('data/desk/colmap_data/right/cameras.txt')

    # Read the existing point cloud data
    colmap_pcd = o3d.io.read_point_cloud('data/desk/colmap_data/right/points.ply')
    colmap_pcd.paint_uniform_color([0.5, 0.5, 0.5])

    # Create visualizer and add geometries
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Change the point size
    render_option = vis.get_render_option()
    render_option.point_size = 0.01  # Set the point size
    
    # Add point cloud geometry to the visualizer
    vis.add_geometry(colmap_pcd)

    # Load the camera parameters for the 'head', 'left' and 'right' cameras
    intrin_head = camera_params['head']['intrinsic']
    intrin_left = camera_params['left']['intrinsic']
    intrin_right = camera_params['right']['intrinsic']

    # Load the extrinsic parameters for all files
    # with open(f'data/desk/camera_extrinsic/head_extrinsic.pkl', 'rb') as f:
    #     head_extrin = pickle.load(f)
    # with open(f'data/desk/camera_extrinsic/left_extrinsic.pkl', 'rb') as f:
    #     left_extrin = pickle.load(f)
    # with open(f'data/desk/camera_extrinsic/right_extrinsic.pkl', 'rb') as f:
    #     right_extrin = pickle.load(f)

    with open(f'head_extrinsics.pkl', 'rb') as f:
        head_extrin = pickle.load(f)
    with open(f'left_extrinsics.pkl', 'rb') as f:
        left_extrin = pickle.load(f)
    with open(f'right_extrinsics.pkl', 'rb') as f:
        right_extrin = pickle.load(f)

    total_frames = len(head_extrin)

    # Sort the keys (file names) in the extrinsic parameters dictionary
    sorted_filenames = sorted(head_extrin.keys())

    # Initialize the first lines geometry
    lines_head = lines_left = lines_right = None

    # Initialize a counter for saving frames of visualized hand joints
    counter = 0
    
    for filename in tqdm(sorted_filenames):
        if filename.endswith('.jpg'):
            # Generate the corresponding filenames for the 'left' and 'right' views
            left_filename = filename.replace('head', 'left')
            right_filename = filename.replace('head', 'right')

            # If there is a corresponding 'head' frame, update the 'head' lines
            if filename in head_extrin:
                new_lines_head = plot_camera_position(intrin_head, head_extrin[filename], img_width=3840, img_height=2160)
                if lines_head is not None:
                    vis.remove_geometry(lines_head)
                vis.add_geometry(new_lines_head)
                lines_head = new_lines_head

            # If there is a corresponding 'left' frame, update the 'left' lines
            if left_filename in left_extrin:
                new_lines_left = plot_camera_position(intrin_left, left_extrin[left_filename], img_width=3840, img_height=2160)
                if lines_left is not None:
                    vis.remove_geometry(lines_left)
                vis.add_geometry(new_lines_left)
                lines_left = new_lines_left

            # If there is a corresponding 'right' frame, update the 'right' lines
            if right_filename in right_extrin:
                new_lines_right = plot_camera_position(intrin_right, right_extrin[right_filename], img_width=3840, img_height=2160)
                if lines_right is not None:
                    vis.remove_geometry(lines_right)
                vis.add_geometry(new_lines_right)
                lines_right = new_lines_right

            # Update the view
            field_of_view, front, lookat, up, zoom = load_view("data/views/view_desk.json")
            ctr = vis.get_view_control()
            ctr.change_field_of_view(field_of_view)
            ctr.set_front(front)
            ctr.set_lookat(lookat)
            ctr.set_up(up)
            ctr.set_zoom(zoom)
            if counter != 0:
                vis.capture_screen_image(f"output_camera/{counter:05}.jpg")
            counter += 1
            vis.poll_events()
            vis.update_renderer()

    ## For rotating video 
    # # Create output directory if not exists
    # output_dir = "camera_rotate"
    # os.makedirs(output_dir, exist_ok=True)

    # # Rotation and capture
    # for i in tqdm(range(360)):
    #     ctr.rotate(10.0, 0.0) # Rotate 1 degree
    #     vis.poll_events()
    #     vis.update_renderer()
    #     vis.capture_screen_image(f"{output_dir}/frame_{i}.png", True)  # Save frame

    # Constructing the ffmpeg command
    ffmpeg_command = f"ffmpeg -r 30 -i output_camera/%5d.jpg -vcodec libx264 -pix_fmt yuv420p fridge.mp4"
    subprocess.call(ffmpeg_command, shell=True)