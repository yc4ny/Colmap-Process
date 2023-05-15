import argparse 
import numpy as np 
import open3d as o3d 
import json
import os
import subprocess
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Visualize 3D Hand in 3D Space.')
parser.add_argument('--capture', type=str, required=True, help='Name of the captured data.', default = "desk")
args = parser.parse_args()

def load_view(json_path): 
    with open(json_path, 'r') as f:
        view_params = json.load(f)
    trajectory = view_params["trajectory"][0]
    return trajectory['field_of_view'], trajectory['front'], trajectory['lookat'], trajectory['up'], trajectory['zoom']

def main():
    ply_file_path = f'data/{args.capture}/colmap_data/sparse/0/points.ply'
    colmap_pcd = o3d.io.read_point_cloud(ply_file_path)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Scene', width=3840, height=2160)
    vis.add_geometry(colmap_pcd)
    field_of_view, front, lookat, up, zoom = load_view(f"data/views/view_{args.capture}.json")
    ctr = vis.get_view_control()
    ctr.change_field_of_view(field_of_view)
    ctr.set_front(front)
    ctr.set_lookat(lookat)
    ctr.set_up(up)
    ctr.set_zoom(zoom)

    # Create output directory if not exists
    output_dir = f'output_rotate/{args.capture}'
    os.makedirs(output_dir, exist_ok=True)

    # Rotation and capture
    for i in tqdm(range(360)):
        ctr.rotate(10.0, 0.0) # Rotate 1 degree
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(f"{output_dir}/frame_{i}.png", True)  # Save frame

    # Constructing the ffmpeg command
    ffmpeg_command = f"ffmpeg -r 30 -i {output_dir}/frame_%d.png -vcodec libx264 -pix_fmt yuv420p {args.capture}.mp4"
    subprocess.call(ffmpeg_command, shell=True)

if __name__ == "__main__":
    main()
