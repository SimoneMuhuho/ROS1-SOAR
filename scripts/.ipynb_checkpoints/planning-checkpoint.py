#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from nav_msgs.srv import GetMap         # Service to fetch static occupancy grid map
from geometry_msgs.msg import PoseStamped  # Message type for robot pose
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Node container
# -----------------------------------------------------------
class Node:
    def __init__(self, mx, my):
        self.mx = mx                  # X coordinate in map pixels
        self.my = my                  # Y coordinate in map pixels
        self.neighbors = []           # List of connected nodes (edges)

# -----------------------------------------------------------
# Global Planner Node
# -----------------------------------------------------------
class GlobalPlannerNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node("global_planner_manual_connections")

        # Load the static map from ROS service
        self.map = self.load_map()

        # Convert the map into a NumPy grid, store resolution and origin
        self.grid, self.res, self.origin = self.process_map(self.map)

        # Create nodes in a 4x4 grid and a mapping from coordinates to node objects
        self.nodes, self.coord_to_node = self.create_nodes(4, 4)

        # Build the hardcoded tree connections
        self.build_manual_tree()

        # Subscribe to robot pose topic to know where the robot is
        self.robot_pose = None
        rospy.Subscriber("/robot_pose", PoseStamped, self.robot_pose_callback)

        # Visualize the map, tree, and robot position
        self.visualize()

    # ---------------- Map Loading ----------------
    def load_map(self):
        rospy.wait_for_service("static_map")   # Wait until the service is available
        get_map = rospy.ServiceProxy("static_map", GetMap)  # Create service proxy
        return get_map().map                  # Call the service and return the map

    def process_map(self, map_msg):
        # Extract width, height, resolution, and origin of the map
        w, h = map_msg.info.width, map_msg.info.height
        res = map_msg.info.resolution
        ox, oy = map_msg.info.origin.position.x, map_msg.info.origin.position.y

        # Convert ROS map data (1D list) to 2D NumPy array
        grid = np.array(map_msg.data).reshape((h, w))

        # Convert to binary: 0 = free, 1 = wall
        grid = (grid != 0).astype(int)
        return grid, res, (ox, oy)

    # ---------------- Node Creation ----------------
    def create_nodes(self, cols, rows):
        nodes = []
        coord_to_node = {}
        ox, oy = self.origin
        res = self.res

        # Convert block coordinates (bx, by) to map pixel coordinates (mx, my)
        def block_to_map(bx, by):
            mx = int((bx * 1.0 - ox) / res)
            my = int((by * 1.0 - oy) / res)
            return mx, my

        # Create a grid of nodes
        for y in range(rows):
            for x in range(cols):
                mx, my = block_to_map(x, y)   # Map coordinates
                n = Node(mx, my)              # Create Node object
                nodes.append(n)               # Add to node list
                coord_to_node[(x, y)] = n    # Store in dict for easy lookup

        return nodes, coord_to_node

    # ---------------- Hardcoded Tree ----------------
    def build_manual_tree(self):
        # Convenience variable
        c = self.coord_to_node

        # Hardcoded connections between nodes to form the tree
        c[(0,0)].neighbors.append(c[(1,0)])
        c[(1,0)].neighbors.append(c[(1,1)])
        c[(1,1)].neighbors.append(c[(0,1)])
        c[(1,1)].neighbors.append(c[(1,2)])
        c[(1,2)].neighbors.append(c[(1,3)])
        c[(1,3)].neighbors.append(c[(0,3)])
        c[(0,3)].neighbors.append(c[(0,2)])
        c[(1,2)].neighbors.append(c[(2,2)])
        c[(2,2)].neighbors.append(c[(2,3)])
        c[(2,2)].neighbors.append(c[(3,2)])
        c[(3,2)].neighbors.append(c[(3,1)])
        c[(3,1)].neighbors.append(c[(3,0)])
        c[(3,1)].neighbors.append(c[(2,1)])
        c[(2,1)].neighbors.append(c[(2,0)])
        c[(3,2)].neighbors.append(c[(3,3)])

    # ---------------- Robot Pose Callback ----------------
    def robot_pose_callback(self, msg):
        # Store the latest robot pose from /robot_pose topic
        self.robot_pose = msg.pose

    # ---------------- Visualization ----------------
    def visualize(self):
        # Create a figure
        plt.figure(figsize=(8,8))

        # Show map in grayscale (walls dark, free space light)
        plt.imshow(self.grid, cmap="Greys", origin="lower")

        # Draw tree edges in blue
        for n in self.nodes:
            for nb in n.neighbors:
                plt.plot([n.mx, nb.mx], [n.my, nb.my], 'dodgerblue', linewidth=2)

        # Draw nodes as medium blue circles
        xs = [n.mx for n in self.nodes]
        ys = [n.my for n in self.nodes]
        plt.scatter(xs, ys, s=150, c='mediumblue', label='Nodes')

        # Draw robot as a red star if pose is known
        if self.robot_pose:
            rx = int((self.robot_pose.position.x - self.origin[0]) / self.res)
            ry = int((self.robot_pose.position.y - self.origin[1]) / self.res)
            plt.scatter(rx, ry, s=200, c='red', marker='*', label='Robot')

        # Add a legend without duplicates
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        # Equal axis scaling for correct aspect
        plt.axis('equal')
        plt.title("Manual Node Tree with Robot Position")
        plt.show()


# ---------------- Main ----------------
if __name__ == "__main__":
    node = GlobalPlannerNode()  # Instantiate the planner node
    rospy.spin()                # Keep the ROS node alive to receive messages
