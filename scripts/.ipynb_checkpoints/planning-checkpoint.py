#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from nav_msgs.srv import GetMap
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import numpy as np
import matplotlib.pyplot as plt
import tf

# -----------------------------------------------------------
# Node container class
# Represents a graph node mapped to a cell in the map
# -----------------------------------------------------------
class Node:
    """
    @brief Represents a graph node corresponding to a map cell.

    Each node stores its map-space coordinates and its neighbors (edges).
    """

    def __init__(self, mx, my):
        """
        @brief Initialize a Node.

        @param mx Map-space x-coordinate (column index)
        @param my Map-space y-coordinate (row index)
        """
        self.mx = mx
        self.my = my
        self.neighbors = []

    def __repr__(self):
        """
        @brief String representation for debugging.

        @return String "mx,my"
        """
        return f"{self.mx},{self.my}"


# -----------------------------------------------------------
# Global Planner Node
# -----------------------------------------------------------
class GlobalPlannerNode:
    """
    @brief Main ROS node for global path planning and visualization.

    Handles map loading, graph creation, path planning (DFS/A*),
    pose publishing, and visualization.
    """

    def __init__(self):
        """
        @brief Initialize the GlobalPlannerNode.

        Sets up ROS publishers, subscribers, loads the map,
        builds the graph nodes, and prepares the manual tree.
        """
        rospy.init_node("global_planner_manual_connections")

        # Map and grid
        self.map = self.load_map()
        self.grid, self.res, self.origin = self.process_map(self.map)

        # Nodes and coordinate lookup
        self.nodes, self.coord_to_node = self.create_nodes(4, 4)
        self.build_manual_tree()

        # Publishers
        self.path_pub = rospy.Publisher('/global_planner/path', Path, queue_size=10)
        self.pose_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)

        # Robot pose
        self.robot_pose = None
        rospy.Subscriber("/robot_pose", PoseStamped, self.robot_pose_callback)

        # TF listener
        listener = tf.TransformListener()

    # ---------------- Map Loading ----------------
    def load_map(self):
        """
        @brief Retrieve the static occupancy map from ROS.

        @return nav_msgs/OccupancyGrid containing the static map
        """
        rospy.wait_for_service("static_map")
        get_map = rospy.ServiceProxy("static_map", GetMap)
        return get_map().map

    def process_map(self, map_msg):
        """
        @brief Convert OccupancyGrid message into a 2D binary grid.

        @param map_msg OccupancyGrid message
        @return tuple(grid, resolution, origin) where
                grid: 2D NumPy array (0=free, 1=obstacle)
                resolution: meters per grid cell
                origin: (ox, oy) world origin coordinates
        """
        w, h = map_msg.info.width, map_msg.info.height
        res = map_msg.info.resolution
        ox, oy = map_msg.info.origin.position.x, map_msg.info.origin.position.y

        grid = np.array(map_msg.data).reshape((h, w))
        grid = (grid != 0).astype(int)
        return grid, res, (ox, oy)

    # ---------------- Node Creation ----------------
    def create_nodes(self, cols, rows):
        """
        @brief Create graph nodes arranged in a grid.

        @param cols Number of columns
        @param rows Number of rows
        @return tuple(nodes, coord_to_node)
                nodes: list of Node objects
                coord_to_node: dict mapping (col,row) to Node
        """
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
        """
        @brief Create edges between nodes based on row/column connectivity.

        Checks visibility and adds bidirectional edges.
        """
        for node in self.nodes:
            neighbors = self.get_row_or_column_neighbors(node)
            neighbors = [n for n in neighbors if n != node]
            if not neighbors:
                continue

            row_neighbors = [n for n in neighbors if n.my == node.my]
            if row_neighbors:
                closest_x = min(row_neighbors, key=lambda n: abs(n.mx - node.mx))
                if self.has_horizontal_edge(node, closest_x):
                    self.add_edge(node, closest_x)

            col_neighbors = [n for n in neighbors if n.mx == node.mx]
            if col_neighbors:
                closest_y = min(col_neighbors, key=lambda n: abs(n.my - node.my))
                if self.has_vertical_edge(node, closest_y):
                    self.add_edge(node, closest_y)

    def get_row_or_column_neighbors(self, node):
        """
        @brief Return nodes sharing either row or column with the given node.

        @param node Node to check
        @return list of neighboring Node objects
        """
        return [n for n in self.nodes if (n.mx == node.mx) ^ (n.my == node.my)]

    def has_horizontal_edge(self, node1, node2):
        """
        @brief Check if a horizontal edge exists (no obstacles in between).

        @param node1 Node on the row
        @param node2 Node on the row
        @return True if clear, False otherwise
        """
        assert node1.my == node2.my
        y = node1.my
        for x in range(min(node1.mx, node2.mx) + 1, max(node1.mx, node2.mx)):
            if self.grid[y, x] == 1:
                return False
        return True

    def has_vertical_edge(self, node1, node2):
        """
        @brief Check if a vertical edge exists (no obstacles in between).

        @param node1 Node on the column
        @param node2 Node on the column
        @return True if clear, False otherwise
        """
        assert node1.mx == node2.mx
        x = node1.mx
        for y in range(min(node1.my, node2.my) + 1, max(node1.my, node2.my)):
            if self.grid[y, x] == 1:
                return False
        return True

    # ---------------- Coordinate Conversion ----------------
    def block_to_map(self, bx, by):
        """
        @brief Convert grid block coordinates to map indices.

        @param bx Block x-coordinate
        @param by Block y-coordinate
        @return tuple(mx, my) map indices
        """
        ox, oy = self.origin
        res = self.res
        mx = int((bx - ox) / res)
        my = int((by - oy) / res)
        return mx, my

    def map_to_block(self, mx, my):
        """
        @brief Convert map indices to world coordinates.

        @param mx Map x-index
        @param my Map y-index
        @return tuple(bx, by) world coordinates
        """
        ox, oy = self.origin
        res = self.res
        bx = (mx + 0.5) * res + ox
        by = (my + 0.5) * res + oy
        return bx, by

    # ---------------- Graph Management ----------------
    def add_edge(self, a, b):
        """
        @brief Add a bidirectional edge between two nodes.

        @param a Node
        @param b Node
        """
        if b not in a.neighbors:
            a.neighbors.append(b)
        if a not in b.neighbors:
            b.neighbors.append(a)

    def build_manual_tree(self):
        """
        @brief Placeholder for building a manual tree of edges.
        """
        c = self.coord_to_node

    # ---------------- Robot Pose ----------------
    def robot_pose_callback(self, msg):
        """
        @brief Update robot pose and trigger visualization.

        @param msg PoseStamped ROS message
        """
        self.robot_pose = msg.pose
        self.visualize()

    def make_pose(self, x, y, yaw=0.0, frame="map"):
        """
        @brief Create a PoseStamped message.

        @param x X-coordinate
        @param y Y-coordinate
        @param yaw Rotation around Z-axis in radians
        @param frame Coordinate frame
        @return PoseStamped message
        """
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = frame
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0
        q = tf.transformations.quaternion_from_euler(0, 0, yaw)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        return pose

    def publish_path(self, path):
        """
        @brief Publish a path as a ROS Path message.

        @param path list of Node objects representing the path
        """
        msg = Path()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"

        for node in path:
            wx, wy = self.map_to_block(node.mx, node.my)
            pose = self.make_pose(wx, wy)
            msg.poses.append(pose)

        self.path_pub.publish(msg)

    # ---------------- Path Planning ----------------
    def dfs(self, start_node, goal_node):
        """
        @brief Depth-First Search from start to goal.

        @param start_node Node to start from
        @param goal_node Goal node
        @return list of Node objects representing path or None
        """
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
        if found:
            self.publish_path(path)
            for node in path:
                bx, by = self.map_to_block(node.mx, node.my)
                self.pose_pub.publish(self.make_pose(bx, by))
            return path
        else:
            return None

    def manhattan_h(self, node, goal_node):
        """
        @brief Manhattan distance heuristic.

        @param node Current node
        @param goal_node Goal node
        @return Integer distance
        """
        return abs(goal_node.mx - node.mx) + abs(goal_node.my - node.my)

    def sort_h(self, nodes):
        """
        @brief Select node with lowest heuristic cost to goal.

        @param nodes list of Node objects
        @return Node with lowest heuristic
        """
        lowest_cost = float('inf')
        chosen = None
        goal = self.make_goal_node()
        for node in nodes:
            current = self.manhattan_h(node, goal)
            if current < lowest_cost:
                lowest_cost = current
                chosen = node
        return chosen

    def parse_goal(self, goal):
        """
        @brief Parse a ROS parameter goal string.

        @param goal string in format "x,y"
        @return tuple (gx, gy) as integers
        """
        gx, gy = map(int, goal.split(','))
        if not (0 <= gx <= 3 and 0 <= gy <= 3):
            raise ValueError("Goal invalid")
        return gx, gy

    def make_goal_node(self):
        """
        @brief Create goal node from ROS parameter or default.

        @return Node object representing goal
        """
        goal = rospy.get_param('~goal')
        gx, gy = 3, 3
        try:
            gx, gy = self.parse_goal(goal)
        except Exception:
            print('Goal invalid, using default 3,3')
        return self.coord_to_node[(gx, gy)]

    def a_star(self, start_node, goal_node):
        """
        @brief Perform A* pathfinding from start to goal.

        @param start_node Node to start from
        @param goal_node Goal Node
        @return list of Node objects representing path or None
        """
        open_set = set([start_node])
        closed_set = set()
        parent = {}

        goal = goal_node
        found = False

        while open_set:
            current = self.sort_h(open_set)
            if current == goal:
                found = True
                break
            open_set.remove(current)
            closed_set.add(current)
            for neighbor in current.neighbors:
                if neighbor in closed_set:
                    continue
                if neighbor not in open_set:
                    open_set.add(neighbor)
                    parent[neighbor] = current

        if found:
            path = []
            node = goal
            while node != start_node:
                path.append(node)
                node = parent[node]
            path.append(start_node)
            path.reverse()

            self.publish_path(path)
            for node in path:
                bx, by = self.map_to_block(node.mx, node.my)
                self.pose_pub.publish(self.make_pose(bx, by))
            return path
        else:
            return None

    # ---------------- Visualization ----------------
    def visualize(self):
        """
        @brief Visualize map, nodes, edges, robot position, and path.

        Builds graph edges, performs A* pathfinding, and plots the results.
        """
        self.create_edges()
        rx = round(self.robot_pose.position.x)
        ry = round(self.robot_pose.position.y)
        start_node = self.coord_to_node.get((rx, ry), None)
        goal_node = self.make_goal_node()
        path = self.a_star(start_node, goal_node) if start_node else None

        plt.rcParams['figure.figsize'] = [7, 7]
        fig, ax = plt.subplots()
        wall_positions = np.argwhere(self.grid == 1)
        ax.scatter(wall_positions[:, 1], wall_positions[:, 0], c='darkblue', alpha=1.0, s=36, label="Walls")

        node_positions = np.array([[n.mx, n.my] for n in self.nodes])
        ax.scatter(node_positions[:, 0], node_positions[:, 1], c='mediumblue', alpha=1.0, s=64, label="Nodes")

        if path:
            path_positions = np.array([[n.mx, n.my] for n in path])
            ax.scatter(path_positions[:, 0], path_positions[:, 1], c='green', alpha=1.0, s=64, label="A* Path")
            for idx in range(1, len(path_positions)):
                x0, y0 = path_positions[idx - 1]
                x1, y1 = path_positions[idx]
                ax.plot([x0, x1], [y0, y1], c='green', linewidth=2)

        rx_px = int((self.robot_pose.position.x - self.origin[0]) / self.res)
        ry_px = int((self.robot_pose.position.y - self.origin[1]) / self.res)
        ax.scatter(rx_px, ry_px, c='red', s=225, marker='*', label="Robot Position")

        for n in self.nodes:
            for nb in n.neighbors:
                ax.plot([n.mx, nb.mx], [n.my, nb.my], c='dodgerblue', linewidth=1)

        ax.set_xlabel("X-Coordinate [pixels]")
        ax.set_ylabel("Y-Coordinate [pixels]")
        ax.set_title("Found Path from Robot Position to Goal")
        ax.set_xticks(np.arange(0, self.grid.shape[1], 1))
        ax.set_yticks(np.arange(0, self.grid.shape[0], 1))
        ax.grid(True)
        ax.set_aspect('equal')
        ax.legend(loc='upper left', framealpha=0.8)
        plt.show()


# ---------------- Main ----------------
if __name__ == "__main__":
    """
    @brief Run the global planner node.
    """
    node = GlobalPlannerNode()
    rospy.spin()
