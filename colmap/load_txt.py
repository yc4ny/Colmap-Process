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

def read_cameras_txt(file_path, camera_id):
    """Reads camera parameters from a text file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith('#'):  # Skip comment lines
            continue
        camera_info = line.strip().split()
        if int(camera_info[0]) == camera_id:
            return [float(val) for val in camera_info[4:]]

    raise ValueError(f"Camera with id {camera_id} not found in file.")

def read_images_txt(file_path):
    """Reads image data from a text file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    image_data = []
    for i in range(4, len(lines), 2):
        image_info = lines[i].strip().split()
        img_id, qw, qx, qy, qz, tx, ty, tz, cam_id, name = image_info
        image_data.append({
            'id': int(img_id),
            'quaternion': [float(qw), float(qx), float(qy), float(qz)],
            'translation': [float(tx), float(ty), float(tz)],
            'camera_id': int(cam_id),
            'name': name
        })
    return image_data

def read_points3D_txt(file_path):
    """Reads 3D points data from a text file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    points3D = []
    for line in lines[3:]:
        data = line.strip().split()
        point_id, x, y, z = map(float, data[:4])
        points3D.append([point_id, x, y, z])
    return points3D