#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file maze_escape.py
@brief Localization and visualization node using laser scans and kNN.

This ROS node estimates the robot position in a known occupancy grid map
by matching laser scan data against a kNN classifier trained on the map.
The estimated pose is published and visualized in real time.
"""

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


# -----------------------------------------------------------
# ROS utilities
# -----------------------------------------------------------

def init_ros():
    """
    @brief Initialize the ROS node.

    Creates the ROS node named @c mazeEscape and enables ROS logging.
    This function must be called before any ROS communication is used.
    """
    rospy.init_node('mazeEscape', anonymous=True)
    rospy.loginfo("Node started: mazeEscape")


def get_map() -> OccupancyGrid:
    """
    @brief Retrieve the static occupancy grid map.

    Waits for the @c /static_map service and requests the map.

    @return The received occupancy grid map.
    """
    rospy.wait_for_service('static_map')
    get_map_srv = rospy.ServiceProxy('static_map', GetMap)
    return get_map_srv().map


def scan_callback(msg: LaserScan):
    """
    @brief Process incoming laser scan data.

    Converts valid laser scan ranges from polar coordinates into
    Cartesian coordinates in the robot frame and stores them globally.

    @param msg Incoming LaserScan message.
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
    @brief Publish the estimated robot pose.

    Publishes the robot position as a PoseStamped message in the @c map
    coordinate frame with zero orientation.

    @param robot_pose Tuple containing the robot position (x, y).
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


# -----------------------------------------------------------
# Coordinate transforms
# -----------------------------------------------------------

def map_to_world(mx, my, rec_map):
    """
    @brief Convert map indices to world coordinates.

    @param mx Map x-index (column).
    @param my Map y-index (row).
    @param rec_map Occupancy grid map metadata.
    @return World coordinates (x, y) in meters.
    """
    res = rec_map.info.resolution
    ox = rec_map.info.origin.position.x
    oy = rec_map.info.origin.position.y
    return mx * res + ox, my * res + oy


def world_to_map(wx, wy, rec_map):
    """
    @brief Convert world coordinates to map indices.

    @param wx World x-coordinate in meters.
    @param wy World y-coordinate in meters.
    @param rec_map Occupancy grid map metadata.
    @return Map indices (mx, my).
    """
    res = rec_map.info.resolution
    ox = rec_map.info.origin.position.x
    oy = rec_map.info.origin.position.y
    return (wx - ox) / res, (wy - oy) / res


def transform_scan(scan_points, robot_pose):
    """
    @brief Transform laser scan points into the world frame.

    Applies a planar rotation and translation based on the robot pose.

    @param scan_points Nx2 array of scan points in robot frame.
    @param robot_pose Tuple (x, y, theta) describing robot pose.
    @return Nx2 array of scan points in world coordinates.
    """
    x, y, theta = robot_pose
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return (R @ scan_points.T).T + np.array([x, y])


# -----------------------------------------------------------
# Localization logic
# -----------------------------------------------------------

def push_away(scan_points, distance=0.10):
    """
    @brief Push laser scan points outward from the robot.

    Helps avoid numerical issues caused by points too close to the origin.

    @param scan_points Nx2 array of laser scan points.
    @param distance Distance to push points outward in meters.
    @return Modified scan points.
    """
    r = np.linalg.norm(scan_points, axis=1)
    r[r == 0] = 1e-6
    return scan_points * ((r + distance) / r)[:, None]


def localize_robot(scan_points, knn, free_cells):
    """
    @brief Estimate the robot position using scan-to-map matching.

    Evaluates translated laser scans against a kNN map classifier and
    selects the position with the highest number of wall hits.

    @param scan_points Laser scan points in robot frame.
    @param knn Trained kNN classifier.
    @param free_cells List of possible robot positions.
    @return Estimated robot position (x, y).
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


# -----------------------------------------------------------
# Map & ML preparation
# -----------------------------------------------------------

def build_map_array(rec_map):
    """
    @brief Convert an occupancy grid into a NumPy array.

    @param rec_map Received occupancy grid map.
    @return 2D NumPy array representation of the map.
    """
    w = rec_map.info.width
    h = rec_map.info.height
    return np.array(rec_map.data, dtype=np.int8).reshape((h, w))


def prepare_knn_data(map_array, rec_map):
    """
    @brief Generate training data for the kNN classifier.

    Converts each map cell into world coordinates and labels it as
    free space or obstacle.

    @param map_array 2D occupancy grid array.
    @param rec_map Occupancy grid map metadata.
    @return Tuple (X, y, free_cells).
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
    @brief Train a k-Nearest Neighbors classifier.

    @param X World coordinates of map cells.
    @param y Occupancy labels.
    @return Trained kNN classifier.
    """
    knn = neighbors.KNeighborsClassifier(n_neighbors=1)
    knn.fit(X, y)
    rospy.loginfo("kNN model fitted successfully.")
    return knn


# -----------------------------------------------------------
# Visualization
# -----------------------------------------------------------

def visualize_knn(knn, X):
    """
    @brief Visualize the learned kNN decision boundary.

    Displays free space and obstacles as classified by the model.
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
    @brief Main localization and visualization loop.

    Continuously estimates the robot pose, publishes it, and visualizes
    laser scans and robot position on the map.
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


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

def main():
    """
    @brief Program entry point.

    Initializes ROS, loads the map, trains the classifier, sets up
    communication, and starts the visualization loop.
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
