# MIT License
#
# Copyright (c) 2023 Yonwoo Choi, Seoul National University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import open3d as o3d
import numpy as np 
import argparse 
import json 

parser = argparse.ArgumentParser(description='Visualize 3D Hand in 3D Space.')
parser.add_argument('--capture', type=str, required=True, help='Name of the captured data.', default = "desk")
args = parser.parse_args()

def load_view(json_path): 
    with open(json_path, 'r') as f:
        view_params = json.load(f)
    trajectory = view_params["trajectory"][0]
    return trajectory['field_of_view'], trajectory['front'], trajectory['lookat'], trajectory['up'], trajectory['zoom']

def main():
    scene_path = f'data/{args.capture}/colmap_data/sparse/0/points.ply'
    scene_pcd = o3d.io.read_point_cloud(scene_path)
    # scene_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Grey pointcloud
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