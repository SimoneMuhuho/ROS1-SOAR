#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.srv import GetMap
from nav_msgs.msg import OccupancyGrid
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import copy

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

    # Polar → Cartesian
    xs = ranges * np.cos(angles)
    ys = ranges * np.sin(angles)

    scan_data = np.vstack((xs, ys)).T


# --- Map <-> World coordinate conversions ---
def map_to_world(mx, my, recMap):
    res = recMap.info.resolution
    ox = recMap.info.origin.position.x
    oy = recMap.info.origin.position.y
    wx = mx * res + (ox + res / 2.0)
    wy = my * res + (oy + res / 2.0)
    return wx, wy


def world_to_map(wx, wy, recMap):
    res = recMap.info.resolution
    ox = recMap.info.origin.position.x
    oy = recMap.info.origin.position.y
    mx = (wx - ox) / res
    my = (wy - oy) / res
    return mx, my


# --- Start ROS node ---
rospy.init_node('mazeEscape', anonymous=True)
rospy.loginfo("Node started: mazeEscape")

# --- Load the map ---
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
#map_array = np.flipud(map_array)  # flip vertically to align with world frame

# --- Plot initial map ---
plt.figure(figsize=(8, 8))
plt.imshow(map_array, cmap='gray', origin='lower')
plt.title('Loaded Map from ROS')
plt.xlabel("x (map cells)")
plt.ylabel("y (map cells)")
plt.pause(1)

# --- Subscribe to LaserScan ---
scan_sub = rospy.Subscriber('/scan', LaserScan, scan_callback)
rospy.loginfo("Subscribed to /scan topic.")

# --- Visualization loop ---
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))

while not rospy.is_shutdown():
    if scan_data is not None:
        # Convert LaserScan (world coordinates) to map pixel coordinates
        xs_world = scan_data[:, 0]
        ys_world = scan_data[:, 1]
        xs_map, ys_map = world_to_map(xs_world, ys_world, recMap)

        ax.clear()
        ax.imshow(map_array, cmap='gray', origin='lower')
        ax.scatter(xs_map, ys_map, c='r', s=8, label='LaserScan (map frame)')
        ax.set_title('Map + LaserScan Overlay')
        ax.set_xlabel('Map X (pixels)')
        ax.set_ylabel('Map Y (pixels)')
        ax.legend()
        plt.pause(0.1)

plt.ioff()
plt.show()
