import open3d as o3d
import numpy as np
import pickle
import time

# Load extrinsics and point cloud
with open('extrinsics.pkl', 'rb') as f:
    extrinsics = pickle.load(f)

pcd = o3d.io.read_point_cloud('scene.ply')

# Visualize point cloud and camera locations sequentially
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)

for key in sorted(extrinsics.keys()):
    extrinsic_matrix = extrinsics[key]
    rotation = extrinsic_matrix[:, :3]
    translation = extrinsic_matrix[:, 3]

    # Compute camera location in world coordinates
    camera_location = -rotation.T @ translation

    # Create a sphere to represent the camera location
    camera_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=2)
    camera_sphere.paint_uniform_color([1, 0, 0])  # Red color
    camera_sphere.translate(camera_location)

    # Visualize camera location
    vis.add_geometry(camera_sphere)
    vis.poll_events()
    vis.update_renderer()

    # Wait for a moment before moving to the next camera location
    time.sleep(0.3)

    # Remove the camera sphere
    vis.remove_geometry(camera_sphere)

vis.destroy_window()
