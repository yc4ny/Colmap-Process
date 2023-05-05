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

import cv2
import argparse

# Function to handle mouse click events
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        height, width, channels = param.shape
        y = height - y  # shift y-coordinate to bottom-left origin
        print("Coordinates: ({}, {})".format(x, y))

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Display an image and print coordinates of left mouse button clicks.')
    parser.add_argument('--image_path', help='Path to the input image file')

    args = parser.parse_args()

    # Read input image
    image = cv2.imread(args.image_path)

    # Create a window to display the image
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", 600, 400)  # set window size

    # Show the image in the window
    cv2.imshow("Image", image)

    # Set up mouse callback function to handle click events
    cv2.setMouseCallback("Image", click_event, param=image)

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()