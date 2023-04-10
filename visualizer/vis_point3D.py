import open3d as o3d
import argparse
import numpy as np 

# Convert quaternion to rotation matrix
def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    R = np.zeros((3, 3))
    R[0, 0] = 1 - 2 * qy ** 2 - 2 * qz ** 2
    R[0, 1] = 2 * qx * qy - 2 * qz * qw
    R[0, 2] = 2 * qx * qz + 2 * qy * qw
    R[1, 0] = 2 * qx * qy + 2 * qz * qw
    R[1, 1] = 1 - 2 * qx ** 2 - 2 * qz ** 2
    R[1, 2] = 2 * qy * qz - 2 * qx * qw
    R[2, 0] = 2 * qx * qz - 2 * qy * qw
    R[2, 1] = 2 * qy * qz + 2 * qx * qw
    R[2, 2] = 1 - 2 * qx ** 2 - 2 * qy ** 2

    return R

# Read image extrinsic parameters and 2D keypoints from the file
def read_images(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    extrinsic_parameters = {}
    keypoints = {}
    for idx, line in enumerate(lines):
        if line.startswith('#') or line.strip() == '':
            continue
        if idx % 2 == 0:
            line_parts = line.strip().split()
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name = line_parts
            rotation = quaternion_to_rotation_matrix(float(qw), float(qx), float(qy), float(qz))
            translation = np.array([float(tx), float(ty), float(tz)]).reshape(3, 1)
            extrinsic = np.hstack((rotation, translation))
            extrinsic_parameters[name] = extrinsic
        else:
            coords = line.strip().split()
            coords = [float(x) for x in coords]
            coords = np.array(coords).reshape(-1, 3)[:, :2]
            keypoints[name] = coords

    sorted_names = sorted(extrinsic_parameters.keys())
    extrinsics_array = np.stack([extrinsic_parameters[name] for name in sorted_names], axis=0)
    keypoints_array = [keypoints[name] for name in sorted_names]

    return extrinsics_array, keypoints_array

def create_camera_coordinate_frame(extrinsic, size=1.0):
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :4] = extrinsic
    coordinate_frame.transform(transformation_matrix)
    return coordinate_frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize 3D point cloud using Open3D')
    parser.add_argument('--input', type=str, help='path to .ply file')
    args = parser.parse_args()

    # Read point cloud from file
    pcd = o3d.io.read_point_cloud(args.input)
    # extrinsic, _ = read_images("colmap/data_undistort/sparse/left/images.txt")


    # # Create camera coordinate frames from extrinsics
    # camera_frames = [create_camera_coordinate_frame(e, size=0.3) for e in extrinsic]

    # Visualize point cloud and camera frames
    # o3d.visualization.draw_geometries([pcd, *camera_frames])
    o3d.visualization.draw_geometries([pcd])