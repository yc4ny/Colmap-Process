import open3d as o3d
import json

# Load the scene point cloud
scene_ply_path = "colmap_data/points.ply"
pcd = o3d.io.read_point_cloud(scene_ply_path)

# Load the 3D joints JSON data
json_file = "output_3d_joints.json"
with open(json_file, 'r') as f:
    joints = json.load(f)

# Function to create geometries for hand joints
def create_hand_geometry(hand_data, color):
    points = []
    lines = []
    
    for i, finger in enumerate(hand_data):
        for j, joint in enumerate(finger):
            points.append(joint)
            if j > 0:
                lines.append([len(points) - 2, len(points) - 1])
                
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector([color] * len(points))
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
    
    return point_cloud, line_set

# Create geometries for left and right hands
left_points, left_lines = create_hand_geometry(joints["left"], [1, 0, 0])
right_points, right_lines = create_hand_geometry(joints["right"], [0, 1, 0])

# Visualize the scene and hand joints
o3d.visualization.draw_geometries([pcd, left_points, left_lines, right_points, right_lines])
