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

import re
import numpy as np
import quaternion
import pickle
import argparse

def extract_extrinsics(images_txt):
    extrinsics = {}

    with open(images_txt, 'r') as file:
        lines = file.readlines()

    for i in range(len(lines)):
        line = lines[i].strip()

        if not line or line.startswith('#'):
            continue

        if re.match(r'right_\d{5}\.jpg', line.split()[-1]):
            image_data = line.split()
            qw, qx, qy, qz = map(float, image_data[1:5])
            tx, ty, tz = map(float, image_data[5:8])
            
            rotation = quaternion.as_rotation_matrix(quaternion.quaternion(qw, qx, qy, qz))
            translation = np.array([tx, ty, tz]).reshape(3, 1)

            extrinsic_matrix = np.hstack((rotation, translation))
            extrinsics[line.split()[-1]] = extrinsic_matrix

    return extrinsics

images_txt_path = 'colmap_data/right/images.txt'
extrinsics = extract_extrinsics(images_txt_path)

with open('right_extrinsics.pkl', 'wb') as file:
    pickle.dump(extrinsics, file)

print(f"Saved extrinsics for {len(extrinsics)} images in extrinsics.pkl")