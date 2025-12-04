#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from nav_msgs.srv import GetMap
from geometry_msgs.msg import PoseStamped
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Node container
# -----------------------------------------------------------
class Node:
    def __init__(self, mx, my):
        self.mx = mx
        self.my = my
        self.neighbors = []

    def __repr__(self):
        return str(self.mx) + ',' + str(self.my)

# -----------------------------------------------------------
# Global Planner Node
# -----------------------------------------------------------
class GlobalPlannerNode:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("global_planner_manual_connections")

        # Load the static map
        self.map = self.load_map()

        # Convert the map into a NumPy grid
        self.grid, self.res, self.origin = self.process_map(self.map)

        # Create nodes and mapping
        self.nodes, self.coord_to_node = self.create_nodes(4, 4)

        # Build manual tree
        self.build_manual_tree()

        # Robot pose
        self.robot_pose = None

        # Subscribe to robot pose
        rospy.Subscriber("/robot_pose", PoseStamped, self.robot_pose_callback)

    # ---------------- Map Loading ----------------
    def load_map(self):
        rospy.wait_for_service("static_map")
        get_map = rospy.ServiceProxy("static_map", GetMap)
        return get_map().map

    def process_map(self, map_msg):
        w, h = map_msg.info.width, map_msg.info.height
        res = map_msg.info.resolution
        ox, oy = map_msg.info.origin.position.x, map_msg.info.origin.position.y
        grid = np.array(map_msg.data).reshape((h, w))
        grid = (grid != 0).astype(int)  # 0=free, 1=wall
        return grid, res, (ox, oy)

    # ---------------- Node Creation ----------------
    def create_nodes(self, cols, rows):
        nodes = []
        coord_to_node = {}
        for y in range(rows):
            for x in range(cols):
                mx, my = self.block_to_map(x, y)
                n = Node(mx, my)
                nodes.append(n)
                coord_to_node[(x, y)] = n
        return nodes, coord_to_node

    def create_edges(self):
        for node in self.nodes:
            neighbors = self.get_row_or_column_neighbors(node)
            # Exclude the node itself
            neighbors = [n for n in neighbors if n != node]

            if not neighbors:
                continue

            # Closest neighbor in x (same row)
            row_neighbors = [n for n in neighbors if n.my == node.my]
            if row_neighbors:
                closest_x = min(row_neighbors, key=lambda n: abs(n.mx - node.mx))
                if self.has_horizontal_edge(node, closest_x):
                    self.add_edge(node, closest_x)

            # Closest neighbor in y (same column)
            col_neighbors = [n for n in neighbors if n.mx == node.mx]
            if col_neighbors:
                closest_y = min(col_neighbors, key=lambda n: abs(n.my - node.my))
                if self.has_vertical_edge(node, closest_y):
                    self.add_edge(node, closest_y)

    def get_row_or_column_neighbors(self, node):
        # Return all nodes sharing x or y, but not both
        return [n for n in self.nodes if (n.mx == node.mx) ^ (n.my == node.my)]

    def has_horizontal_edge(self, node1, node2):
        assert node1.my == node2.my, "Nodes must be in same row"
        y = node1.my
        for x in range(min(node1.mx, node2.mx) + 1, max(node1.mx, node2.mx)):
            if self.grid[y, x] == 1:
                return False
        return True

    def has_vertical_edge(self, node1, node2):
        assert node1.mx == node2.mx, "Nodes must be in same column"
        x = node1.mx
        for y in range(min(node1.my, node2.my) + 1, max(node1.my, node2.my)):
            if self.grid[y, x] == 1:
                return False
        return True

    def block_to_map(self, bx, by):
        ox, oy = self.origin
        res = self.res
        mx = int((bx - ox) / res)
        my = int((by - oy) / res)
        return mx, my

    # ---------------- Add Bidirectional Edge ----------------
    def add_edge(self, a, b):
        if b not in a.neighbors:
            a.neighbors.append(b)
        if a not in b.neighbors:
            b.neighbors.append(a)

    # ---------------- Hardcoded Tree ----------------
    def build_manual_tree(self):
        c = self.coord_to_node
        '''self.add_edge(c[(0,0)], c[(1,0)])
        self.add_edge(c[(1,0)], c[(1,1)])
        self.add_edge(c[(1,1)], c[(0,1)])
        self.add_edge(c[(1,1)], c[(1,2)])
        self.add_edge(c[(1,2)], c[(1,3)])
        self.add_edge(c[(1,3)], c[(0,3)])
        self.add_edge(c[(0,3)], c[(0,2)])
        self.add_edge(c[(1,2)], c[(2,2)])
        self.add_edge(c[(2,2)], c[(2,3)])
        self.add_edge(c[(2,2)], c[(3,2)])
        self.add_edge(c[(3,2)], c[(3,1)])
        self.add_edge(c[(3,1)], c[(3,0)])
        self.add_edge(c[(3,1)], c[(2,1)])
        self.add_edge(c[(2,1)], c[(2,0)])
        self.add_edge(c[(3,2)], c[(3,3)])'''

    # ---------------- Robot Pose Callback ----------------
    def robot_pose_callback(self, msg):
        self.robot_pose = msg.pose
        self.visualize()

    # ---------------- DFS ----------------
    def dfs(self, start_node, goal_node):
        visited = set()
        path = []

        def recurse(node):
            if node in visited:
                return False
            visited.add(node)
            path.append(node)
            if node == goal_node:
                return True
            for child in node.neighbors:
                if recurse(child):
                    return True
            path.pop()
            return False

        found = recurse(start_node)
        return path if found else None

    # ---------------- Visualization ----------------
    def visualize(self):
        self.create_edges()
        
        # Determine starting node from robot position
        rx = round(self.robot_pose.position.x)
        ry = round(self.robot_pose.position.y)
        start_node = self.coord_to_node.get((rx, ry), None)
        goal = rospy.get_param('~goal')
        gx, gy = 3, 3
        try:
            gx, gy = list(map(int, goal.split(',')))
        except Exception:
            print('Goal invalid, using default 3,3')
        goal_node = self.coord_to_node[(gx,gy)]
        path = self.dfs(start_node, goal_node) if start_node else None

        # ------------------- Figure Setup -------------------
        plt.rcParams['figure.figsize'] = [7, 7]
        fig, ax = plt.subplots()

        # ------------------- Walls -------------------
        wall_positions = np.argwhere(self.grid == 1)
        ax.scatter(
            wall_positions[:, 1], wall_positions[:, 0],
            c='darkblue', alpha=1.0, s=6**2, label="Walls"
        )

        # ------------------- Nodes -------------------
        node_positions = np.array([[n.mx, n.my] for n in self.nodes])
        ax.scatter(
            node_positions[:, 0],
            node_positions[:, 1],
            c='mediumblue', alpha=1.0, s=8**2, label="Nodes"
        )

        # ------------------- DFS Path -------------------
        if path:
            path_positions = np.array([[n.mx, n.my] for n in path])
            ax.scatter(
                path_positions[:, 0],
                path_positions[:, 1],
                c='green', alpha=1.0, s=8**2, label="DFS Path"
            )
            for idx in range(1, len(path_positions)):
                x0, y0 = path_positions[idx-1]
                x1, y1 = path_positions[idx]
                ax.plot([x0, x1], [y0, y1], c='green', linewidth=2)

        # ------------------- Robot -------------------
        rx_px = int((self.robot_pose.position.x - self.origin[0]) / self.res)
        ry_px = int((self.robot_pose.position.y - self.origin[1]) / self.res)
        ax.scatter(rx_px, ry_px, c='red', s=15**2, marker='*', label="Robot Position")

        # ------------------- Tree Edges -------------------
        for n in self.nodes:
            for nb in n.neighbors:
                ax.plot([n.mx, nb.mx], [n.my, nb.my], c='dodgerblue', linewidth=1)

        # ------------------- Axes & Grid -------------------
        ax.set_xlabel("X-Coordinate [pixels]")
        ax.set_ylabel("Y-Coordinate [pixels]")
        ax.set_title("Found Path from Robot Position to Goal")
        ax.set_xticks(np.arange(0, self.grid.shape[1], 1))
        ax.set_yticks(np.arange(0, self.grid.shape[0], 1))
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.legend(loc='upper left', framealpha=0.8)  # up
        ax.set_aspect('equal')
        plt.show()


# ---------------- Main ----------------
if __name__ == "__main__":
    node = GlobalPlannerNode()
    rospy.spin()
