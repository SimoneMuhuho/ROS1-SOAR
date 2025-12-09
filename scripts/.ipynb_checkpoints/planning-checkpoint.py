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
# Node container
# -----------------------------------------------------------
class Node:
    def __init__(self, map_x, map_y):
        self.map_x = map_x
        self.map_y = map_y
        self.neighbors = []

    def __repr__(self):
        return f"{self.map_x},{self.map_y}"


# -----------------------------------------------------------
# Global Planner Node
# -----------------------------------------------------------
class GlobalPlannerNode:
    def __init__(self):
        rospy.init_node("global_planner_manual_connections")

        # Load the static map
        self.map_message = self.load_map()

        # Convert map message to a grid
        self.occupancy_grid, self.resolution, self.map_origin = self.process_map(self.map_message)

        # Create graph nodes
        self.nodes, self.grid_coord_to_node = self.create_nodes(num_columns=4, num_rows=4)

        # Build graph connections
        self.create_edges()

        # Publishers
        self.path_publisher = rospy.Publisher('/global_planner/path', Path, queue_size=10)
        self.goal_pose_publisher = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)

        # Robot pose
        self.robot_pose = None
        rospy.Subscriber("/robot_pose", PoseStamped, self.robot_pose_callback)

        # TF listener
        self.tf_listener = tf.TransformListener()

    # ---------------- Map Loading ----------------
    def load_map(self):
        rospy.wait_for_service("static_map")
        get_map_service = rospy.ServiceProxy("static_map", GetMap)
        return get_map_service().map

    def process_map(self, map_message):
        width = map_message.info.width
        height = map_message.info.height
        resolution = map_message.info.resolution

        origin_x = map_message.info.origin.position.x
        origin_y = map_message.info.origin.position.y

        occupancy_array = np.array(map_message.data).reshape((height, width))
        occupancy_array = (occupancy_array != 0).astype(int)   # 0 = free, 1 = occupied

        return occupancy_array, resolution, (origin_x, origin_y)

    # ---------------- Node Creation ----------------
    def create_nodes(self, num_columns, num_rows):
        nodes = []
        grid_coord_to_node = {}

        for row in range(num_rows):
            for col in range(num_columns):
                map_x, map_y = self.block_to_map(col, row)
                node = Node(map_x, map_y)
                nodes.append(node)
                grid_coord_to_node[(col, row)] = node

        return nodes, grid_coord_to_node

    def create_edges(self):
        for node in self.nodes:
            potential_neighbors = self.get_row_or_column_neighbors(node)

            # Exclude itself
            potential_neighbors = [nbr for nbr in potential_neighbors if nbr != node]

            if not potential_neighbors:
                continue

            # --- closest neighbor horizontally ---
            horizontal_neighbors = [nbr for nbr in potential_neighbors if nbr.map_y == node.map_y]
            if horizontal_neighbors:
                closest_horizontal = min(horizontal_neighbors, key=lambda nbr: abs(nbr.map_x - node.map_x))
                if self.path_clear_horizontal(node, closest_horizontal):
                    self.add_edge(node, closest_horizontal)

            # --- closest neighbor vertically ---
            vertical_neighbors = [nbr for nbr in potential_neighbors if nbr.map_x == node.map_x]
            if vertical_neighbors:
                closest_vertical = min(vertical_neighbors, key=lambda nbr: abs(nbr.map_y - node.map_y))
                if self.path_clear_vertical(node, closest_vertical):
                    self.add_edge(node, closest_vertical)

    def get_row_or_column_neighbors(self, node):
        # match either X OR Y, but not both
        return [
            nbr for nbr in self.nodes
            if (nbr.map_x == node.map_x) ^ (nbr.map_y == node.map_y)
        ]

    def path_clear_horizontal(self, node_a, node_b):
        assert node_a.map_y == node_b.map_y
        y = node_a.map_y

        for x in range(min(node_a.map_x, node_b.map_x) + 1,
                       max(node_a.map_x, node_b.map_x)):
            if self.occupancy_grid[y, x] == 1:
                return False

        return True

    def path_clear_vertical(self, node_a, node_b):
        assert node_a.map_x == node_b.map_x
        x = node_a.map_x

        for y in range(min(node_a.map_y, node_b.map_y) + 1,
                       max(node_a.map_y, node_b.map_y)):
            if self.occupancy_grid[y, x] == 1:
                return False

        return True

    # ---------------- Coordinate Conversion ----------------
    def block_to_map(self, block_x, block_y):
        origin_x, origin_y = self.map_origin
        resolution = self.resolution

        map_x = int((block_x - origin_x) / resolution)
        map_y = int((block_y - origin_y) / resolution)
        return map_x, map_y

    def map_to_block(self, map_x, map_y):
        origin_x, origin_y = self.map_origin
        resolution = self.resolution

        block_x = (map_x + 0.5) * resolution + origin_x
        block_y = (map_y + 0.5) * resolution + origin_y
        return block_x, block_y

    # ---------------- Add Bidirectional Edge ----------------
    def add_edge(self, node_a, node_b):
        if node_b not in node_a.neighbors:
            node_a.neighbors.append(node_b)
        if node_a not in node_b.neighbors:
            node_b.neighbors.append(node_a)

    # ---------------- Robot Pose Callback ----------------
    def robot_pose_callback(self, pose_message):
        self.robot_pose = pose_message.pose
        self.visualize()

    # ---------- Create Goal Node ------------------
    def get_goal_node(self):
        goal_param = rospy.get_param('~goal')
        goal_col, goal_row = 3, 3

        try:
            goal_col, goal_row = list(map(int, goal_param.split(',')))
        except Exception:
            print("Invalid goal parameter, using default (3,3)")

        return self.grid_coord_to_node[(goal_col, goal_row)]

    # ---------- Create PoseStamped ------------------
    def make_pose(self, x_world, y_world, yaw=0.0, frame="map"):
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = frame

        pose.pose.position.x = float(x_world)
        pose.pose.position.y = float(y_world)
        pose.pose.position.z = 0.0

        quat = tf.transformations.quaternion_from_euler(0, 0, yaw)
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]

        return pose

    # ---------- Publish Path ------------------
    def publish_path(self, node_path):
        path_message = Path()
        path_message.header.stamp = rospy.Time.now()
        path_message.header.frame_id = "map"

        for node in node_path:
            world_x, world_y = self.map_to_block(node.map_x, node.map_y)
            pose = self.make_pose(world_x, world_y)
            path_message.poses.append(pose)

        self.path_publisher.publish(path_message)

    # ---------------- DFS Algorithm ----------------
    def dfs(self, start_node, goal_node):
        visited_nodes = set()
        current_path = []

        def traverse(current_node):
            if current_node in visited_nodes:
                return False

            visited_nodes.add(current_node)
            current_path.append(current_node)

            if current_node == goal_node:
                return True

            for neighbor in current_node.neighbors:
                if traverse(neighbor):
                    return True

            current_path.pop()
            return False

        found = traverse(start_node)

        if found:
            for node in current_path:
                world_x, world_y = self.map_to_block(node.map_x, node.map_y)
                self.publish_path(current_path)
                self.goal_pose_publisher.publish(self.make_pose(world_x, world_y))
            return current_path

        return None

    # ---------------- Visualization ----------------
    def visualize(self):
        # compute start node
        robot_map_x = round(self.robot_pose.position.x)
        robot_map_y = round(self.robot_pose.position.y)

        start_node = self.grid_coord_to_node.get((robot_map_x, robot_map_y))
        goal_node = self.get_goal_node()

        node_path = self.dfs(start_node, goal_node) if start_node else None

        plt.rcParams['figure.figsize'] = [7, 7]
        fig, ax = plt.subplots()

        # Walls
        wall_coords = np.argwhere(self.occupancy_grid == 1)
        ax.scatter(wall_coords[:, 1], wall_coords[:, 0], c='darkblue', s=36, label="Walls")

        # Node points
        graph_coords = np.array([[n.map_x, n.map_y] for n in self.nodes])
        ax.scatter(graph_coords[:, 0], graph_coords[:, 1], c='mediumblue', s=64, label="Nodes")

        # Path found
        if node_path:
            path_coords = np.array([[n.map_x, n.map_y] for n in node_path])
            ax.scatter(path_coords[:, 0], path_coords[:, 1], c='green', s=64, label="DFS Path")

            for i in range(1, len(path_coords)):
                x0, y0 = path_coords[i - 1]
                x1, y1 = path_coords[i]
                ax.plot([x0, x1], [y0, y1], c='green', linewidth=2)

        # Robot
        robot_grid_x = int((self.robot_pose.position.x - self.map_origin[0]) / self.resolution)
        robot_grid_y = int((self.robot_pose.position.y - self.map_origin[1]) / self.resolution)
        ax.scatter(robot_grid_x, robot_grid_y, c='red', s=225, marker='*', label="Robot")

        # Edges
        for n in self.nodes:
            for neighbor in n.neighbors:
                ax.plot([n.map_x, neighbor.map_x], [n.map_y, neighbor.map_y], c='dodgerblue', linewidth=1)

        ax.set_xlabel("X [map pixels]")
        ax.set_ylabel("Y [map pixels]")
        ax.set_title("Path from Robot to Goal")
        ax.grid(True)
        ax.legend()
        ax.set_aspect('equal')
        plt.show()


# ---------------- Main ----------------
if __name__ == "__main__":
    node = GlobalPlannerNode()
    rospy.spin()
