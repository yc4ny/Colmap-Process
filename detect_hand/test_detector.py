import json
from cv2 import FONT_HERSHEY_SIMPLEX
import numpy as np
import cv2

def main():
    connections = [[0,1], [0,5], [0,17],[1,2], [2,3], [3,4], [0,5], [5,6],[6,7], [7,8],[5,9], [9,10], [10,11], [11,12], [9,13],
                   [13,14], [14,15], [15,16], [13,17], [17,18], [18,19], [19,20]]
    
    with open("detect_hand/left/left_00002_joints.json", 'r') as file:
        joints = json.load(file)
        left = joints['left']
        right = joints['right']

    img = cv2.imread("preprocessed/left/left_00002.jpg")

    hands = [left[0], right[0]]
    green_color = (0, 255, 0)
    red_color = (0, 0, 255)

    # Combine left and right hand loops
    for hand in hands:
        # Precompute coordinates and colors for drawing circles and lines
        coords = [(int(point[0]), int(point[1])) for point in hand]
        lines = [(coords[i], coords[j]) for i, j in connections if [i, j] in connections or [j, i] in connections]

        # Use list comprehension to draw circles
        [cv2.circle(img, coord, 15, green_color, -1) for coord in coords]

        # Use list comprehension to draw lines
        [cv2.line(img, line[0], line[1], red_color, 3) for line in lines]

    cv2.imwrite("detect_hand/test_joint.jpg", img)

if __name__ == "__main__":
    main()
