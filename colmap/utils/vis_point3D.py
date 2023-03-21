import open3d as o3d
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize 3D point cloud using Open3D')
    parser.add_argument('--input', type=str, help='path to .ply file')
    args = parser.parse_args()

    # Read point cloud from file
    pcd = o3d.io.read_point_cloud(args.input)

    # Visualize point cloud
    o3d.visualization.draw_geometries([pcd])