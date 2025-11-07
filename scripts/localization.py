#!/usr/bin/env python3
# --- Setup ---
# %matplotlib inline

import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.srv import GetMap
from nav_msgs.msg import OccupancyGrid
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import copy

# --- Helper method for retrieving the map ---
def getMap() -> OccupancyGrid:
    """Loads map from the /static_map service."""
    rospy.wait_for_service('static_map')
    get_map = rospy.ServiceProxy('static_map', GetMap)
    recMap = get_map()
    return recMap.map


scan_data = None

def scan_callback(msg: LaserScan):
    global scan_data
    # Convert polar to Cartesian
    angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
    ranges = np.array(msg.ranges)

    # Filter out invalid readings
    valid = (ranges > msg.range_min) & (ranges < msg.range_max)
    ranges = ranges[valid]
    angles = angles[valid]

    # Convert to x, y (laser frame)
    xs = ranges * np.cos(angles)
    ys = ranges * np.sin(angles)

    scan_data = np.vstack((xs, ys)).T


# --- Start ROS node ---
rospy.init_node('mazeEscape', anonymous=True)

# --- Load the map ---
recMap = getMap()
print("Map received!")

# --- Convert to numpy image ---
width = recMap.info.width   # retrivies height (y axis) and width (x axis) 
height = recMap.info.height #

map_array = np.array(recMap.data).reshape((height, width)) #concat of funnction that transform the reMap (functionaly a list of occupancy) into a 2D array of occupancy
map_array = np.flipud(map_array) #flip the map to make the origin point 0.0 match with our map

plt.figure(figsize=(8, 8))
plt.imshow(map_array, cmap='gray') #imshow allows Matplotlib to draw the 2D Numpy array, the option gray set the color sccale to the gray one where occupied blocks are black fee ones are free
plt.title('Loaded Map from ROS')

scan_sub = rospy.Subscriber('/scan', LaserScan, scan_callback)

# --- Visualization loop ---
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))

while not rospy.is_shutdown():
    if scan_data is not None:
        ax.clear()
        ax.imshow(map_array, cmap='gray')
        ax.scatter(scan_data[:, 0], scan_data[:, 1], c='r', s=10, label='LaserScan')
        ax.set_title('Map + LaserScan')
        ax.legend()
        plt.pause(0.1)

plt.ioff()

plt.show()

