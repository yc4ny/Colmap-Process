import numpy as np 
import cv2
import pickle
import argparse
import glob
import os

def main(pkl_folder, img_folder, output_folder):
    connections = [
        [0, 1],[1, 2],[2, 3],[3, 4],[0, 5],[5, 6],[6, 7],[7, 8],[0, 9],[9, 10],[10, 11],[11, 12],
        [0, 13],[13, 14],[14, 15],[15, 16],[0, 17],[17, 18],[18, 19],[19, 20]]

    pkl_files = sorted(glob.glob(os.path.join(pkl_folder, '*.pkl')))

    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            joint = pickle.load(f)
            left_hand_data = joint['pred_output_list'][0]['left_hand']
            right_hand_data = joint['pred_output_list'][0]['right_hand']

            left_joint = left_hand_data.get('pred_joints_img')
            right_joint = right_hand_data.get('pred_joints_img')

        base_name = os.path.splitext(os.path.basename(pkl_file))[0].replace('_prediction_result', '')
        img_file = os.path.join(img_folder, f'{base_name}.jpg')
        img = cv2.imread(img_file)

        # Skip if 'pred_joints_img' key doesn't exist
        if left_joint is not None:
            # Draw the joints and the connections for the left hand
            for connection in connections:
                cv2.line(img, tuple(left_joint[connection[0],:2].astype(int)), tuple(left_joint[connection[1],:2].astype(int)), (0,255,0), 5)
            for point in left_joint:
                cv2.circle(img, tuple(point[:2].astype(int)),6, (0,0,255), thickness=-1, lineType=cv2.FILLED)

        if right_joint is not None:
            # Draw the joints and the connections for the right hand
            for connection in connections:
                cv2.line(img, tuple(right_joint[connection[0],:2].astype(int)), tuple(right_joint[connection[1],:2].astype(int)), (0,255,0), 5)
            for point in right_joint:
                cv2.circle(img, tuple(point[:2].astype(int)), 6, (12,0,255), thickness=-1, lineType=cv2.FILLED)

        # Save the image
        output_file = os.path.join(output_folder, f'{base_name}.jpg')
        cv2.imwrite(output_file, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', required=False, help='Folder containing .pkl files', default = "data/desk/frankmocap_joints")
    parser.add_argument('--images', required=False, help='Folder containing corresponding images', default = "data/desk/img/undistort_head")
    parser.add_argument('--output', required=False, help='Output folder to save images with drawn hand joints', default = "frankmocap_desk")
    args = parser.parse_args()

    main(args.pkl, args.images, args.output)
