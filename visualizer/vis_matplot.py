import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the JSON data
with open("output_3d_joints.json", "r") as f:
    data = json.load(f)

# Extract left and right joint coordinates
left_joints = data["left"]
right_joints = data["right"]

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot left and right joint coordinates
for i, (left_joint, right_joint) in enumerate(zip(left_joints, right_joints)):
    left_xs, left_ys, left_zs = zip(*left_joint)
    right_xs, right_ys, right_zs = zip(*right_joint)

    ax.scatter(left_xs, left_ys, left_zs, c="r", marker="o", label=f"Left {i+1}")
    ax.scatter(right_xs, right_ys, right_zs, c="b", marker="o", label=f"Right {i+1}")

# Set axis labels
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Set the legend and display the plot
ax.legend()
plt.show()
