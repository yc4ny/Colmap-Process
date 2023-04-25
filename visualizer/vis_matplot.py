import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_3d_joints_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['3D_Joints']

def visualize_3d_joints(joints_3D):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    unique_joints = []
    for point in joints_3D:
        if point not in unique_joints:
            unique_joints.append(point)
            ax.scatter(point[0], point[1], point[2], c='r', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    
def main():
    file_path = 'output.json'
    joints_3D = load_3d_joints_from_json(file_path)
    visualize_3d_joints(joints_3D)

if __name__ == "__main__":
    main()
