import open3d as o3d
import numpy as np 

def main():
    scene_path = 'gopro_capture/fridge/colmap_data/sparse/0/points.ply'
    scene_pcd = o3d.io.read_point_cloud(scene_path)
    # scene_pcd.paint_uniform_color([0.5, 0.5, 0.5])  
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Scene', width=2400, height=1800)
    vis.add_geometry(scene_pcd)
    vis.run()

if __name__ == "__main__":
    main()