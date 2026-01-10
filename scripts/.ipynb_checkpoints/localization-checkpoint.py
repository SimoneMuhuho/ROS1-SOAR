#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import matplotlib.pyplot as plt
import math

from sensor_msgs.msg import LaserScan
from nav_msgs.srv import GetMap
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Quaternion

from sklearn import neighbors
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib.colors import ListedColormap

import tf.transformations as tft

# Global scan storage (required by ROS callback)

scan_data = None
pose_pub = None


# ROS utilities

def init_ros():
    """
    Initialize the ROS node for this program.

    Creates the ROS node named 'mazeEscape' and enables logging.
    This function must be called before using any ROS features.
    """
    rospy.init_node('mazeEscape', anonymous=True)
    rospy.loginfo("Node started: mazeEscape")


def get_map() -> OccupancyGrid:
    """
    Retrieve the static occupancy grid map from the ROS map server.

    Waits for the '/static_map' service to become available and then
    requests the map.

    Returns:
        OccupancyGrid: The received static map.
    """
    rospy.wait_for_service('static_map')
    get_map_srv = rospy.ServiceProxy('static_map', GetMap)
    return get_map_srv().map


def scan_callback(msg: LaserScan):
    """
    Callback function for the LaserScan subscriber.

    Converts laser scan data from polar coordinates (range, angle)
    into Cartesian coordinates (x, y) in the robot's local frame.
    Invalid range measurements are filtered out.

    The resulting point cloud is stored in the global variable
    'scan_data'.
    """
    global scan_data

    angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
    ranges = np.array(msg.ranges)

    valid = (ranges > msg.range_min) & (ranges < msg.range_max)
    ranges = ranges[valid]
    angles = angles[valid]

    xs = ranges * np.cos(angles)
    ys = ranges * np.sin(angles)

    scan_data = np.vstack((xs, ys)).T


def publish_pose(robot_pose):
    """
    Publish the estimated robot position as a PoseStamped message.

    The pose is published in the 'map' frame with zero orientation
    (yaw = 0), since orientation estimation is not handled here.

    Args:
        robot_pose (tuple): (x, y) position of the robot in world frame.
    """
    px, py = robot_pose

    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "map"

    msg.pose.position.x = px
    msg.pose.position.y = py
    msg.pose.position.z = 0.0

    quat = tft.quaternion_from_euler(0, 0, 0)
    msg.pose.orientation = Quaternion(*quat)

    pose_pub.publish(msg)


# Coordinate transforms

def map_to_world(mx, my, rec_map):
    """
    Convert map grid coordinates (cell indices) to world coordinates.

    Args:
        mx (float): Map x-index (column).
        my (float): Map y-index (row).
        rec_map (OccupancyGrid): Map metadata.

    Returns:
        tuple: (x, y) position in world coordinates (meters).
    """
    res = rec_map.info.resolution
    ox = rec_map.info.origin.position.x
    oy = rec_map.info.origin.position.y
    return mx * res + ox, my * res + oy


def world_to_map(wx, wy, rec_map):
    """
    Convert world coordinates to map grid coordinates.

    Args:
        wx (float): World x-coordinate (meters).
        wy (float): World y-coordinate (meters).
        rec_map (OccupancyGrid): Map metadata.

    Returns:
        tuple: (mx, my) map indices (floating point).
    """
    res = rec_map.info.resolution
    ox = rec_map.info.origin.position.x
    oy = rec_map.info.origin.position.y
    return (wx - ox) / res, (wy - oy) / res


def transform_scan(scan_points, robot_pose):
    """
    Transform laser scan points from the robot frame to world frame.

    Applies a 2D rotation and translation based on the robot pose.

    Args:
        scan_points (np.ndarray): Nx2 array of scan points in robot frame.
        robot_pose (tuple): (x, y, theta) robot pose in world frame.

    Returns:
        np.ndarray: Nx2 array of scan points in world coordinates.
    """
    x, y, theta = robot_pose
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return (R @ scan_points.T).T + np.array([x, y])


# Localization logic

def push_away(scan_points, distance=0.10):
    """
    Push laser scan points slightly outward from the robot.

    This helps prevent points from being too close to the robot,
    improving robustness during localization.

    Args:
        scan_points (np.ndarray): Nx2 laser scan points.
        distance (float): Distance to push points outward (meters).

    Returns:
        np.ndarray: Modified scan points.
    """
    r = np.linalg.norm(scan_points, axis=1)
    r[r == 0] = 1e-6
    return scan_points * ((r + distance) / r)[:, None]


def localize_robot(scan_points, knn, free_cells):
    """
    Estimate the robot position by matching laser scans to the map.

    For each free cell in the map, the scan is translated to that
    location and evaluated using a kNN classifier. The position
    that results in the highest number of wall hits is selected.

    Args:
        scan_points (np.ndarray): Laser scan points in robot frame.
        knn (KNeighborsClassifier): Trained map classifier.
        free_cells (list): List of possible robot positions.

    Returns:
        tuple: Estimated robot position (x, y).
    """
    best_score = -1
    best_pose = None

    scan_points = push_away(scan_points)

    for rx, ry in free_cells:
        transformed = scan_points + np.array([rx, ry])
        predictions = knn.predict(transformed)
        score = np.sum(predictions == 1)

        if score > best_score:
            best_score = score
            best_pose = (rx, ry)

    return best_pose


# Map & ML preparation

def build_map_array(rec_map):
    """
    Convert the OccupancyGrid map data into a 2D NumPy array.

    Args:
        rec_map (OccupancyGrid): Received map.

    Returns:
        np.ndarray: 2D array representing the map.
    """
    w = rec_map.info.width
    h = rec_map.info.height
    return np.array(rec_map.data, dtype=np.int8).reshape((h, w))


def prepare_knn_data(map_array, rec_map):
    """
    Generate training data for the kNN classifier from the map.

    Each map cell is converted into a world coordinate and labeled
    as free space or wall. Free cells are also stored separately
    for localization.

    Args:
        map_array (np.ndarray): 2D occupancy map.
        rec_map (OccupancyGrid): Map metadata.

    Returns:
        tuple: (X, y, free_cells)
            X : World coordinates of all map cells
            y : Corresponding wall/free labels
            free_cells : List of valid robot positions
    """
    h, w = map_array.shape
    X, y, free_cells = [], [], []

    for i in range(h):
        for j in range(w):
            wx, wy = map_to_world(j, i, rec_map)
            label = 1 if map_array[i, j] > 50 else 0

            X.append([wx, wy])
            y.append(label)

            if label == 0:
                free_cells.append((wx, wy))

    return np.array(X), np.array(y), free_cells


def train_knn(X, y):
    """
    Train a k-Nearest Neighbors classifier on the map data.

    Args:
        X (np.ndarray): World coordinates of map cells.
        y (np.ndarray): Occupancy labels.

    Returns:
        KNeighborsClassifier: Trained classifier.
    """
    knn = neighbors.KNeighborsClassifier(n_neighbors=1)
    knn.fit(X, y)
    rospy.loginfo("kNN model fitted successfully.")
    return knn


# Visualization

def visualize_knn(knn, X):
    """
    Visualize the learned kNN decision boundary of the map.

    Displays free space and walls as classified by the trained
    kNN model.
    """
    colours = ListedColormap(["#d3d3d3", "#00008b"])

    plt.rcParams['figure.figsize'] = [5, 5]
    _, ax = plt.subplots()

    DecisionBoundaryDisplay.from_estimator(
        knn,
        X,
        cmap=colours,
        ax=ax,
        response_method="predict",
        plot_method="pcolormesh",
        shading="auto",
        xlabel="X [m]",
        ylabel="Y [m]"
    )

    ax.set_title("Fitted kNN Map Model")
    plt.show(block=False)
    plt.pause(2)


def visualization_loop(map_array, rec_map, knn, free_cells):
    """
    Main runtime loop for localization and visualization.

    Continuously localizes the robot using incoming laser scans,
    publishes the estimated pose, and visualizes the result on
    the map.
    """
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))

    robot_pose = (0.0, 0.0, 0.0)

    while not rospy.is_shutdown():
        if scan_data is None:
            continue

        robot_xy = localize_robot(scan_data, knn, free_cells)
        publish_pose(robot_xy)

        scan_world = transform_scan(scan_data, (robot_xy[0], robot_xy[1], 0.0))
        xs_map, ys_map = world_to_map(scan_world[:, 0], scan_world[:, 1], rec_map)

        ax.clear()
        ax.imshow(map_array, cmap='gray', origin='lower')
        ax.scatter(xs_map, ys_map, c='r', s=8, label='LaserScan')
        ax.scatter(*world_to_map(robot_xy[0], robot_xy[1], rec_map),
                   c='b', s=60, marker='X', label='Robot')

        ax.set_title("Map + LaserScan")
        ax.legend()
        plt.pause(0.1)

    plt.ioff()
    plt.show()


# Main

def main():
    """
    Program entry point.

    Initializes ROS, loads the map, trains the classifier,
    sets up subscribers and publishers, and starts the main loop.
    """
    global pose_pub

    init_ros()

    rec_map = get_map()
    rospy.loginfo("Map received")

    map_array = build_map_array(rec_map)

    X, y, free_cells = prepare_knn_data(map_array, rec_map)
    knn = train_knn(X, y)

    visualize_knn(knn, X)

    rospy.Subscriber('/scan', LaserScan, scan_callback)
    pose_pub = rospy.Publisher("/robot_pose", PoseStamped, queue_size=10)

    rospy.loginfo("Subscribed to /scan")
    visualization_loop(map_array, rec_map, knn, free_cells)


if __name__ == "__main__":
    main()
