import pickle 
import open3d as o3d
import numpy as np 

def visualize_3d_points(left_joints, right_joints, connections, ply_file_path, scale = 10):
    # Load the PLY file
    colmap_pcd = o3d.io.read_point_cloud(ply_file_path)
    colmap_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Grey color for the points from the PLY file

    # Create the PointCloud object for the hand joints
    left = o3d.geometry.PointCloud()
    left.points = o3d.utility.Vector3dVector(left_joints * scale)
    left.paint_uniform_color([1, 0, 0])  # Red color for the hand joints
    left.estimate_normals()

    # Create the PointCloud object for the hand joints
    right = o3d.geometry.PointCloud()
    right.points = o3d.utility.Vector3dVector(right_joints * scale)
    right.paint_uniform_color([1, 0, 0])  # Red color for the hand joints
    right.estimate_normals()

    # Create lines between the connected joints
    left_lines = o3d.geometry.LineSet()
    left_lines.points = left.points
    left_lines.lines = o3d.utility.Vector2iVector(connections)
    left_lines.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in range(len(connections))])  # Blue color for the lines

    # Create lines between the connected joints
    right_lines = o3d.geometry.LineSet()
    right_lines.points = right.points
    right_lines.lines = o3d.utility.Vector2iVector(connections)
    right_lines.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in range(len(connections))])  # Blue color for the lines

    # Create a custom RenderOption
    custom_render_option = o3d.visualization.RenderOption()
    custom_render_option.line_width = 5  # Set the line width to 5 (default is 1)

    # Visualize the point clouds and the lines together with the custom RenderOption
    viewer = o3d.visualization.VisualizerWithKeyCallback()
    viewer.create_window(window_name='PLY and Hand Joints', width=800, height=600)
    viewer.add_geometry(colmap_pcd)
    viewer.add_geometry(left)
    viewer.add_geometry(left_lines)
    viewer.add_geometry(right)
    viewer.add_geometry(right_lines)
    viewer.get_render_option().line_width = custom_render_option.line_width
    viewer.run()
    viewer.destroy_window()
    # # Visualize the point clouds and the lines together
    # o3d.visualization.draw_geometries([colmap_pcd, left, left_lines, right, right_lines], window_name='PLY and Hand Joints', width=800, height=600)


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

    with open('joints_3d/head/head_00000_prediction_result.pkl', 'rb') as f:
        data = pickle.load(f)
    
    left_joints = data['pred_output_list'][0]['left_hand']['pred_joints_smpl']
    right_joints = data['pred_output_list'][0]['right_hand']['pred_joints_smpl']
    print(left_joints.shape)
    
    visualize_3d_points(left_joints,right_joints,  connections, ply_file_path)



if __name__ == "__main__":
    main()