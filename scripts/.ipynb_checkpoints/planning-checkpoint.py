#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from nav_msgs.srv import GetMap
from nav_msgs.msg import OccupancyGrid
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

# --- Coordinate conversions ---
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
    mx = int(round((wx - ox) / res))
    my = int(round((wy - oy) / res))
    return mx, my

# --- Tree Node ---
class TreeNode:
    def __init__(self, mx, my, wx, wy):
        self.mx = mx
        self.my = my
        self.wx = wx
        self.wy = wy
        self.parent = None
        self.children = []
        self.path_to_parent = [(mx, my)]

# --- BFS pathfinder (4-connectivity) ---
def bfs_path(grid, start, end):
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    parent = dict()
    queue = deque([start])
    visited[start[1], start[0]] = True

    while queue:
        x, y = queue.popleft()
        if (x, y) == end:
            path = []
            while (x, y) != start:
                path.append((x, y))
                x, y = parent[(x, y)]
            path.append(start)
            path.reverse()
            return path
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx] and grid[ny, nx] == 0:
                visited[ny, nx] = True
                parent[(nx, ny)] = (x, y)
                queue.append((nx, ny))
    return None

# --- Global Planner Node ---
class GlobalPlannerNode:
    def __init__(self, max_nodes=15, corridor_width=5):
        rospy.init_node("global_planner_diagonal_corridor", anonymous=True)
        rospy.loginfo("Node 2 started: diagonal corridor tree.")

        # --- Load map ---
        self.map = self.load_map()
        self.height = self.map.info.height
        self.width = self.map.info.width
        self.grid = np.array(self.map.data).reshape((self.height, self.width))

        # --- Tree nodes ---
        self.nodes = []
        self.max_nodes = max_nodes
        self.corridor_width = corridor_width

        # --- Build tree along diagonal corridor ---
        self.build_diagonal_corridor_tree()
        self.visualize_tree()

    # --- Load map ---
    def load_map(self) -> OccupancyGrid:
        rospy.wait_for_service("static_map")
        get_map = rospy.ServiceProxy("static_map", GetMap)
        recMap = get_map()
        return recMap.map

    # --- Build tree along diagonal corridor ---
    def build_diagonal_corridor_tree(self):
        # Map indices of start and end
        mx_start, my_start = world_to_map(0.0, 0.0, self.map)
        mx_end, my_end = world_to_map(3.0, 3.0, self.map)

        # Generate all cells along the diagonal
        num_samples = self.max_nodes * 5  # oversample to select free cells
        xs = np.linspace(mx_start, mx_end, num_samples, dtype=int)
        ys = np.linspace(my_start, my_end, num_samples, dtype=int)

        corridor_cells = []
        for x, y in zip(xs, ys):
            # Corridor: include cells around the diagonal
            for dx in range(-self.corridor_width, self.corridor_width+1):
                for dy in range(-self.corridor_width, self.corridor_width+1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        if self.grid[ny, nx] == 0:
                            corridor_cells.append((nx, ny))

        # Remove duplicates
        corridor_cells = list(set(corridor_cells))

        # Randomly pick max_nodes from corridor cells
        selected_cells = random.sample(corridor_cells, min(self.max_nodes, len(corridor_cells)))

        # Sort cells roughly along diagonal for sequential connection
        selected_cells.sort(key=lambda c: c[0]+c[1])

        previous_node = None
        for mx, my in selected_cells:
            wx, wy = map_to_world(mx, my, self.map)
            new_node = TreeNode(mx, my, wx, wy)
            if previous_node:
                path = bfs_path(self.grid, (previous_node.mx, previous_node.my), (mx, my))
                if path:
                    new_node.parent = previous_node
                    previous_node.children.append(new_node)
                    new_node.path_to_parent = path
            self.nodes.append(new_node)
            previous_node = new_node

        rospy.loginfo(f"Diagonal corridor tree built with {len(self.nodes)} nodes.")

    # --- Visualization ---
    def visualize_tree(self):
        plt.figure(figsize=(12,12))
        plt.imshow(self.grid, cmap='gray', origin='lower')

        # Draw edges
        for node in self.nodes:
            if node.path_to_parent:
                px, py = zip(*node.path_to_parent)
                plt.plot(px, py, c='red', linewidth=1)

        # Draw nodes
        xs = [node.mx for node in self.nodes]
        ys = [node.my for node in self.nodes]
        plt.scatter(xs, ys, s=50, c='blue', label='Tree nodes')

        # Draw start and end
        plt.scatter([self.nodes[0].mx], [self.nodes[0].my], s=100, c='green', marker='o', label='Start (0,0)')
        plt.scatter([self.nodes[-1].mx], [self.nodes[-1].my], s=100, c='magenta', marker='x', label='End (3,3)')

        plt.title("Diagonal corridor tree (~15 nodes, safe edges)")
        plt.xlabel("X (grid cells)")
        plt.ylabel("Y (grid cells)")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    node = GlobalPlannerNode(max_nodes=15, corridor_width=5)
    rospy.spin()
