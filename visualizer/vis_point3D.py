import open3d as o3d
import json

# Load the scene point cloud
scene_ply_path = "colmap_data/sparse/0/points3D.ply"
pcd = o3d.io.read_point_cloud(scene_ply_path)

# Load the 3D joints JSON data
json_file = "output.json"
with open(json_file, 'r') as f:
    joints_data = json.load(f)

joints = joints_data["3D_Joints"]

def create_sphere_at_location(location, radius, color):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius)
    sphere.paint_uniform_color(color)
    sphere.translate(location)
    return sphere

# Create spheres for joints
radius = 0.1  # Adjust this value to change the size of the spheres
color = [1, 0, 0]
joint_spheres = [create_sphere_at_location(joint, radius, color) for joint in joints]

# Visualize the scene and joint spheres
geometries = [pcd] + joint_spheres
o3d.visualization.draw_geometries(geometries)
