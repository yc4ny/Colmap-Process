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

"""
Code for generating a filelist of all the images in a folder, which is used as an input to the ffmpeg custom framerate control command

Example ffmpeg command :ffmpeg -f concat -safe 0 -r 30 -i filelist.txt outfilm.mp4 -> Creating a video from images at 30fps
"""

import os
import argparse

def create_image_list(input_folder, output_file):
    image_files = sorted(os.listdir(input_folder))

    with open(output_file, 'w') as f:
        for image_file in image_files:
            if image_file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                f.write(f"file '{os.path.join(input_folder, image_file)}'\n")

def main(): 
    parser = argparse.ArgumentParser(description='Preprocessing mp4 files')
    parser.add_argument('--input', help='directory of folder of images for ffmpeg', default='data/input', required=False)
    parser.add_argument('--output', help='directory of output filelist txt file', default='data/filelist.txt', required=False)
    args = parser.parse_args()
    create_image_list(args.input, args.output)

if __name__ == "__main__":
    main()