import open3d as o3d
import numpy as np 
import argparse 
import json 

def load_view(json_path): 
    with open(json_path, 'r') as f:
        view_params = json.load(f)
    trajectory = view_params["trajectory"][0]
    return trajectory['field_of_view'], trajectory['front'], trajectory['lookat'], trajectory['up'], trajectory['zoom']

def main():

    parser = argparse.ArgumentParser(description='Visualize 3D Hand in 3D Space.')
    parser.add_argument('--capture', type=str, required=True, help='Name of the captured data.', default = "desk")
    args = parser.parse_args()

    scene_path = f'data/{args.capture}/colmap_data/sparse/0/points.ply'
    scene_pcd = o3d.io.read_point_cloud(scene_path)
    # scene_pcd.paint_uniform_color([0.5, 0.5, 0.5])  
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Scene', width=2400, height=1800)
    vis.add_geometry(scene_pcd)

    field_of_view, front, lookat, up, zoom = load_view(f"data/views/view_{args.capture}.json")
    ctr = vis.get_view_control()
    ctr.change_field_of_view(field_of_view)
    ctr.set_front(front)
    ctr.set_lookat(lookat)
    ctr.set_up(up)
    ctr.set_zoom(zoom)

    vis.run()

if __name__ == "__main__":
    main()