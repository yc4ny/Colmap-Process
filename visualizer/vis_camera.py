import open3d as o3d
import numpy as np
import pickle 

def create_camera_marker(intrinsic, extrinsic, size=0.2):
    """
    Creates a LineSet that represents a camera.
    
    Args:
        intrinsic: The camera's intrinsic parameters as a PinholeCameraIntrinsic object.
        extrinsic: The camera's extrinsic parameters as a 4x4 matrix.
        size: The size of the camera marker.
    
    Returns:
        A LineSet that represents the camera.
    """
    fx = intrinsic[0][0]
    fy = intrinsic[1][1]
    cx = intrinsic[0][2]
    cy = intrinsic[1][2]

    # Calculate the 3D points of the frustum corners
    near = size
    far = 2 * size
    x_near = near * cx / fx
    y_near = near * cy / fy
    x_far = far * cx / fx
    y_far = far * cy / fy
    XYZ = np.array([[0, 0, 0],  # Camera center
                    [-x_near, -y_near, near],  # Near plane, bottom-left corner
                    [x_near, -y_near, near],  # Near plane, bottom-right corner
                    [x_near, y_near, near],  # Near plane, top-right corner
                    [-x_near, y_near, near],  # Near plane, top-left corner
                    [-x_far, -y_far, far],  # Far plane, bottom-left corner
                    [x_far, -y_far, far],  # Far plane, bottom-right corner
                    [x_far, y_far, far],  # Far plane, top-right corner
                    [-x_far, y_far, far]])  # Far plane, top-left corner

    # Apply the extrinsic matrix to the points
    XYZ = np.dot(XYZ, extrinsic[:3, :3].T) + extrinsic[:3, 3]

    # Create the lines for the frustum
    lines = [[0, 1], [0, 2], [0, 3], [0, 4],  # Lines from camera center to near plane corners
             [1, 2], [2, 3], [3, 4], [4, 1],  # Near plane edges
             [5, 6], [6, 7], [7, 8], [8, 5],  # Far plane edges
             [1, 5], [2, 6], [3, 7], [4, 8]]  # Lines connecting near and far planes

    # Create a LineSet
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(XYZ),
        lines=o3d.utility.Vector2iVector(lines),
    )

    return line_set

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
    # Create a Visualizer and add the trajectory
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    colmap_pcd = o3d.io.read_point_cloud('data/desk/colmap_data/right/points.ply')
    colmap_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Grey color for the points from the PLY file
    vis.add_geometry(colmap_pcd)
    camera_params = get_camera_params('data/desk/colmap_data/right/cameras.txt')
    with open(f'data/desk/camera_extrinsic/head_extrinsic.pkl', 'rb') as f:
        head_extrin = pickle.load(f)
    with open(f'data/desk/camera_extrinsic/left_extrinsic.pkl', 'rb') as f:
        left_extrin = pickle.load(f)
    with open(f'data/desk/camera_extrinsic/right_extrinsic.pkl', 'rb') as f:
        right_extrin = pickle.load(f)
    extrin_1 = head_extrin['head_00001.jpg']
    intrin_1 = camera_params['head']['intrinsic']
    extrin_2 = left_extrin['left_00001.jpg']
    intrin_2 = camera_params['left']['intrinsic']
    extrin_3  =right_extrin['right_00001.jpg']
    intrin_3 = camera_params['right']['intrinsic']
    # Add the camera marker
    head_marker = create_camera_marker(intrin_1, extrin_1)
    left_marker = create_camera_marker(intrin_2, extrin_2)
    right_marker = create_camera_marker(intrin_3, extrin_3)
    vis.add_geometry(head_marker)
    vis.add_geometry(left_marker)
    vis.add_geometry(right_marker)

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()