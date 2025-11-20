#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##sed -i 's/\r$//' ~/catkin_ws/src/fhtw/ROS1-SOAR/scripts/localization.py
import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.srv import GetMap
from nav_msgs.msg import OccupancyGrid
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay
import math
from geometry_msgs.msg import PoseStamped, Quaternion
import tf.transformations as tft

# --- Retrieve static map ---
def getMap() -> OccupancyGrid:
    """Loads map from the /static_map service."""
    rospy.wait_for_service('static_map')
    get_map = rospy.ServiceProxy('static_map', GetMap)
    recMap = get_map()
    return recMap.map


# --- Global variable for scan data ---
scan_data = None


def scan_callback(msg: LaserScan):
    """Convert LaserScan polar data to Cartesian coordinates (laser frame)."""
    global scan_data

    # Create angle array
    angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
    ranges = np.array(msg.ranges)

    # Filter invalid readings
    valid = (ranges > msg.range_min) & (ranges < msg.range_max)
    ranges = ranges[valid]
    angles = angles[valid]

    # Polar → Cartesian (in robot/laser frame)
    xs = ranges * np.cos(angles)
    ys = ranges * np.sin(angles)

    scan_data = np.vstack((xs, ys)).T


# --- Map <-> World coordinate conversions ---
def map_to_world(mx, my, recMap):
    res = recMap.info.resolution
    ox = recMap.info.origin.position.x
    oy = recMap.info.origin.position.y
    wx = mx * res + ox
    wy = my * res + oy
    return wx, wy


def world_to_map(wx, wy, recMap):
    res = recMap.info.resolution
    ox = recMap.info.origin.position.x
    oy = recMap.info.origin.position.y
    mx = (wx - ox) / res
    my = (wy - oy) / res
    return mx, my


# --- Transform scan points to world frame ---
def transform_scan(scan_points, robot_pose):
    """
    Transform scan points from robot frame to world/map frame.
    robot_pose = (x, y, theta)
    """
    x_r, y_r, theta_r = robot_pose

    c, s = np.cos(theta_r), np.sin(theta_r)
    R = np.array([[c, -s], [s, c]])
    transformed = (R @ scan_points.T).T + np.array([x_r, y_r])
    return transformed


# --- Localization function ---
def localize_robot(scan_points, knn, free_cells):
    best_score = -1
    best_pose = None

    scan_points = push_away(scan_points)
    for rx, ry in free_cells:
        scan_transformed = scan_points + np.array([rx, ry])
        predictions = knn.predict(scan_transformed)
        wall_count = np.sum(predictions == 1)

        if wall_count > best_score:
            best_score = wall_count
            best_pose = (rx, ry)

    return best_pose

def push_away(scan_points, distance = 0.10):
    r = np.linalg.norm(scan_points, axis=1)
    r[r == 0] = 1e-6
    r_new = r + distance
    return scan_points * (r_new / r)[:, None]

def publish_pose(robot_pose):
    pose_msg = PoseStamped()
    pose_msg.header.stamp = rospy.Time.now()
    pose_msg.header.frame_id = "map"
    
    px, py = robot_pose
    
    pose_msg.pose.position.x = px
    pose_msg.pose.position.y = py
    pose_msg.pose.position.z = 0.0
    
    quat = tft.quaternion_from_euler(0, 0, 0)
    pose_msg.pose.orientation = Quaternion(*quat)
    
    pose_pub.publish(pose_msg)


# --- Initialize ROS node ---
rospy.init_node('mazeEscape', anonymous=True)
rospy.loginfo("Node started: mazeEscape")

# --- Load map ---
recMap = getMap()
rospy.loginfo("Map received!")

width = recMap.info.width
height = recMap.info.height
resolution = recMap.info.resolution
origin = recMap.info.origin.position

rospy.loginfo(f"Map width={width}, height={height}, resolution={resolution}")
rospy.loginfo(f"Origin: ({origin.x}, {origin.y})")

# --- Convert map data to 2D numpy array ---
map_array = np.array(recMap.data, dtype=np.int8).reshape((height, width))

# --- Prepare training data for kNN ---
X, y = [], []
free_cells = []

for i in range(height):
    for j in range(width):
        wx, wy = map_to_world(j, i, recMap)
        X.append([wx, wy])
        label = 1 if map_array[i, j] > 50 else 0
        y.append(label)
        if label == 0:
            free_cells.append((wx, wy))

X = np.array(X)
y = np.array(y)

# --- Train kNN model ---
knn = neighbors.KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)
rospy.loginfo("kNN model fitted successfully.")

# --- Visualize trained model ---
colourScheme = {
    "twgrey": "#d3d3d3",   # free space
    "darkblue": "#00008b"  # walls
}

plt.rcParams['figure.figsize'] = [5, 5]
_, ax = plt.subplots()

DecisionBoundaryDisplay.from_estimator(
    knn,
    X,
    cmap=ListedColormap([
        colourScheme["twgrey"],
        colourScheme["darkblue"]
    ]),
    ax=ax,
    response_method="predict",
    plot_method="pcolormesh",
    xlabel="X-Coordinate [m]",
    ylabel="Y-Coordinate [m]",
    shading="auto"
)

ax.set_title("Fitted kNN Model representing the Map")
plt.show(block=False)
plt.pause(2)

# --- Subscribe to LaserScan ---
scan_sub = rospy.Subscriber('/scan', LaserScan, scan_callback)
rospy.loginfo("Subscribed to /scan topic.")

# --- Visualization loop ---
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))

# Assume robot starts at origin facing +x
robot_pose = (0.0, 0.0, 0.0)

starting_positions = []
for i in range(4):
    for j in range(4):
        starting_positions.append((i, j))

pose_pub = rospy.Publisher("/robot_pose", PoseStamped, queue_size=10)

while not rospy.is_shutdown():
    if scan_data is not None:
        # Compute the robot pose using the latest scan
        robot_pose = localize_robot(scan_data, knn, free_cells)
        publish_pose(robot_pose)

        # Transform scan to world coordinates
        scan_world = transform_scan(scan_data, (robot_pose[0], robot_pose[1], 0.0))

        # Convert to map pixel coordinates
        xs_world = scan_world[:, 0]
        ys_world = scan_world[:, 1]
        xs_map, ys_map = world_to_map(xs_world, ys_world, recMap)

        ax.clear()
        ax.imshow(map_array, cmap='gray', origin='lower')
        ax.scatter(xs_map, ys_map, c='r', s=8, label='LaserScan (map frame)')
        ax.scatter(*world_to_map(robot_pose[0], robot_pose[1], recMap),
                   c='b', s=50, marker='X', label='Robot Pose')
        ax.set_title('Map + LaserScan Overlay')
        ax.set_xlabel('Map X (pixels)')
        ax.set_ylabel('Map Y (pixels)')
        ax.legend()
        plt.pause(0.1)

plt.ioff()
plt.show()
