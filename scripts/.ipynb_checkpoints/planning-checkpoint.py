#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from nav_msgs.srv import GetMap
from geometry_msgs.msg import PoseStamped
import numpy as np
import matplotlib.pyplot as plt
from nav_msgs.msg import Path
import tf

# -----------------------------------------------------------
# Node container class
# Represents a graph node mapped to a cell in the map
# -----------------------------------------------------------
class Node:
    def __init__(self, mx, my):
        # Map-space coordinates of the node
        self.mx = mx
        self.my = my
        # List of neighboring nodes (graph edges)
        self.neighbors = []

    def __repr__(self):
        # String representation for debugging and logging
        return str(self.mx) + ',' + str(self.my)

# -----------------------------------------------------------
# Global Planner Node
# Handles map loading, graph creation, path planning, and visualization
# -----------------------------------------------------------
class GlobalPlannerNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node("global_planner_manual_connections")

        # Load the static map from the map server
        self.map = self.load_map()

        # Convert the map message into a NumPy grid
        self.grid, self.res, self.origin = self.process_map(self.map)

        # Create graph nodes and coordinate lookup table
        self.nodes, self.coord_to_node = self.create_nodes(4, 4)

        # Build a manually defined tree (currently empty)
        self.build_manual_tree()
        
        # ROS publishers for path visualization and goal publishing
        self.path_pub = rospy.Publisher('/global_planner/path', Path, queue_size=10)
        self.pose_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        
        # Store the robot's current pose
        self.robot_pose = None

        # Subscribe to the robot pose topic
        rospy.Subscriber("/robot_pose", PoseStamped, self.robot_pose_callback)

        # TF listener (currently unused but initialized)
        listener = tf.TransformListener()

    # ---------------- Map Loading ----------------
    def load_map(self):
        # Wait for the static map service and retrieve the map
        rospy.wait_for_service("static_map")
        get_map = rospy.ServiceProxy("static_map", GetMap)
        return get_map().map

    def process_map(self, map_msg):
        # Extract map dimensions, resolution, and origin
        w, h = map_msg.info.width, map_msg.info.height
        res = map_msg.info.resolution
        ox, oy = map_msg.info.origin.position.x, map_msg.info.origin.position.y

        # Convert map data into a 2D NumPy grid
        grid = np.array(map_msg.data).reshape((h, w))

        # Convert to binary grid: 0 = free, 1 = obstacle
        grid = (grid != 0).astype(int)
        return grid, res, (ox, oy)

    # ---------------- Node Creation ----------------
    def create_nodes(self, cols, rows):
        # Create graph nodes laid out in a grid
        nodes = []
        coord_to_node = {}
        for y in range(rows):
            for x in range(cols):
                # Convert block coordinates to map coordinates
                mx, my = self.block_to_map(x, y)
                n = Node(mx, my)
                nodes.append(n)
                coord_to_node[(x, y)] = n
        return nodes, coord_to_node

    def create_edges(self):
        # Automatically create graph edges based on visibility
        for node in self.nodes:
            neighbors = self.get_row_or_column_neighbors(node)

            # Remove self references
            neighbors = [n for n in neighbors if n != node]
            if not neighbors:
                continue

            # Find closest node in the same row
            row_neighbors = [n for n in neighbors if n.my == node.my]
            if row_neighbors:
                closest_x = min(row_neighbors, key=lambda n: abs(n.mx - node.mx))
                if self.has_horizontal_edge(node, closest_x):
                    self.add_edge(node, closest_x)

            # Find closest node in the same column
            col_neighbors = [n for n in neighbors if n.mx == node.mx]
            if col_neighbors:
                closest_y = min(col_neighbors, key=lambda n: abs(n.my - node.my))
                if self.has_vertical_edge(node, closest_y):
                    self.add_edge(node, closest_y)

    def get_row_or_column_neighbors(self, node):
        # Return nodes that share either the same row or column (but not both)
        return [n for n in self.nodes if (n.mx == node.mx) ^ (n.my == node.my)]

    def has_horizontal_edge(self, node1, node2):
        # Check for obstacles between two nodes on the same row
        assert node1.my == node2.my, "Nodes must be in same row"
        y = node1.my
        for x in range(min(node1.mx, node2.mx) + 1, max(node1.mx, node2.mx)):
            if self.grid[y, x] == 1:
                return False
        return True

    def has_vertical_edge(self, node1, node2):
        # Check for obstacles between two nodes on the same column
        assert node1.mx == node2.mx, "Nodes must be in same column"
        x = node1.mx
        for y in range(min(node1.my, node2.my) + 1, max(node1.my, node2.my)):
            if self.grid[y, x] == 1:
                return False
        return True

    # ---------------- Coordinate Conversion ----------------
    def block_to_map(self, bx, by):
        # Convert block (grid) coordinates to map pixel coordinates
        ox, oy = self.origin
        res = self.res
        mx = int((bx - ox) / res)
        my = int((by - oy) / res)
        return mx, my
        
    def map_to_block(self, mx, my):
        # Convert map pixel coordinates back to world coordinates
        ox, oy = self.origin
        res = self.res
        bx = (mx + 0.5) * res + ox
        by = (my + 0.5) * res + oy
        return bx, by

    # ---------------- Add Bidirectional Edge ----------------
    def add_edge(self, a, b):
        # Add a bidirectional graph connection between two nodes
        if b not in a.neighbors:
            a.neighbors.append(b)
        if a not in b.neighbors:
            b.neighbors.append(a)

    # ---------------- Hardcoded Tree ----------------
    def build_manual_tree(self):
        # Placeholder for manual edge creation
        c = self.coord_to_node

    # ---------------- Robot Pose Callback ----------------
    def robot_pose_callback(self, msg):
        # Update robot pose and trigger visualization
        self.robot_pose = msg.pose
        self.visualize()

    # ---------------- Goal Pose Creator ----------------
    def make_pose(self, x, y, yaw=0.0, frame="map"):
        # Create a PoseStamped message for a given position and orientation
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = frame
    
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0
    
        # Convert yaw angle to quaternion
        q = tf.transformations.quaternion_from_euler(0, 0, yaw)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        return pose

    # ---------------- Path Publisher ----------------
    def publish_path(self, path):
        # Publish a ROS Path message from a list of nodes
        msg = Path()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
    
        for node in path:
            wx, wy = self.map_to_block(node.mx, node.my)
            pose = self.make_pose(wx, wy)
            msg.poses.append(pose)
    
        self.path_pub.publish(msg)

    # ---------------- Depth-First Search ----------------
    def dfs(self, start_node, goal_node):
        # Perform DFS search from start to goal
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
    
        # Publish poses if path is found
        if found:
            for node in path:
                bx, by = self.map_to_block(node.mx, node.my)
                self.publish_path(path)
                self.pose_pub.publish(self.make_pose(bx, by))
            return path
        else:
            return None

    # ---------------- Manhattan Distance Heuristic ----------------
    def manhattan_h(self, node, goal_node):
        # Compute Manhattan distance heuristic
        return abs(goal_node.mx - node.mx) + abs(goal_node.my - node.my)

    def sort_h(self, nodes):
        # Select node with lowest heuristic cost
        lowest_cost = 100000
        chosen = None
        goal = self.make_goal_node()
        for node in nodes:
            current = self.manhattan_h(node, goal)
            if current < lowest_cost:
                lowest_cost = current
                chosen = node
        return chosen

    def parse_goal(self, goal):
        # Parse and validate goal coordinates from ROS parameter
        gx, gy = map(int, goal.split(','))
        if not (0 <= gx <= 3 and 0 <= gy <= 3):
            raise ValueError("Goal invalid")
        return gx, gy

    # ---------------- Dynamic Goal Node Creation ----------------
    def make_goal_node(self):
        # Create goal node from ROS parameter or default
        goal = rospy.get_param('~goal')
        gx, gy = 3, 3
        try:
            gx, gy = self.parse_goal(goal)
        except Exception:
            print('Goal invalid, using default 3,3')
        return self.coord_to_node[(gx, gy)]

    # ---------------- A* Search ----------------
    def a_star(self, start_node, goal_node):
        # Perform A* pathfinding
        open_set = set()
        closed_set = set()
        parent = {}

        open_set.add(start_node)
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

        # Path reconstruction
        if found:
            path = []
            node = goal
            while node != start_node:
                path.append(node)
                node = parent[node]
            path.append(start_node)
            path.reverse()

            # Publish path and poses
            self.publish_path(path)
            for node in path:
                bx, by = self.map_to_block(node.mx, node.my)
                self.pose_pub.publish(self.make_pose(bx, by))

            return path
        else:
            return None

    # ---------------- Visualization ----------------
    def visualize(self):
        # Build edges and visualize the map, nodes, and path
        self.create_edges()
        
        rx = round(self.robot_pose.position.x)
        ry = round(self.robot_pose.position.y)
        start_node = self.coord_to_node.get((rx, ry), None)
        goal_node = self.make_goal_node()
        path = self.a_star(start_node, goal_node) if start_node else None

        plt.rcParams['figure.figsize'] = [7, 7]
        fig, ax = plt.subplots()

        # Draw walls
        wall_positions = np.argwhere(self.grid == 1)
        ax.scatter(wall_positions[:, 1], wall_positions[:, 0],
                   c='darkblue', alpha=1.0, s=6**2, label="Walls")

        # Draw nodes
        node_positions = np.array([[n.mx, n.my] for n in self.nodes])
        ax.scatter(node_positions[:, 0], node_positions[:, 1],
                   c='mediumblue', alpha=1.0, s=8**2, label="Nodes")

        # Draw A* path
        if path:
            path_positions = np.array([[n.mx, n.my] for n in path])
            ax.scatter(path_positions[:, 0], path_positions[:, 1],
                       c='green', alpha=1.0, s=8**2, label="A* Path")
            for idx in range(1, len(path_positions)):
                x0, y0 = path_positions[idx - 1]
                x1, y1 = path_positions[idx]
                ax.plot([x0, x1], [y0, y1], c='green', linewidth=2)

        # Draw robot position
        rx_px = int((self.robot_pose.position.x - self.origin[0]) / self.res)
        ry_px = int((self.robot_pose.position.y - self.origin[1]) / self.res)
        ax.scatter(rx_px, ry_px, c='red', s=15**2, marker='*', label="Robot Position")

        # Draw graph edges
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
    # Instantiate and spin the global planner node
    node = GlobalPlannerNode()
    rospy.spin()
