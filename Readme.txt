ROS1-SOAR: Maze Localization and Global Planning Project 

This project demonstrates a basic autonomous navigation system for a TurtleBot3 robot using ROS1 in simulation. The goal of the project is to show how localization and global planning can be combined to allow a robot to navigate inside a known environment. The system runs in an online JupyterLab ROS environment and uses Gazebo for simulation together with RViz and Matplotlib for visualization.

---HOW IT WORKS---
When the project is launched, a TurtleBot3 robot is spawned in a simulated world along with a preloaded static map. Laser scan data from the robot is used together with the map to estimate the robot’s position. This estimated pose is continuously published so it can be used by other components of the system. Based on the robot’s current position and a predefined goal, a global planner computes a path through the environment and publishes both the path and goal for navigation. The process is visualized to help understand how the robot localizes itself and plans its movement.

The project consists of one launch file and two custom ROS nodes. The launch file starts the Gazebo simulation, loads the map, initializes the TurtleBot3, and runs all required ROS nodes including move_base. The localization node estimates the robot’s position using laser scan data and the static map, while the planning node builds a simple representation of the environment and computes a path from the robot’s position to the goal.
---HOW TO RUN IT---
To run the project in the online JupyterLab environment, make sure ROS1 and the TurtleBot3 packages are properly sourced. First source the ROS and workspace setup files, then set the TurtleBot3 model (for example, burger). The project can then be started using the roslaunch command with the provided launch file

1)open a konsole and after gettng into the launch directory type
	- sed -i 's/\r$//' ~/catkin_ws/src/fhtw/ros1_soar/scripts/localization.py
	- sed -i 's/\r$//' ~/catkin_ws/src/fhtw/ros1_soar/scripts/planning.py
These are needed to make the "\n" and "/" otherwise known as the "new line" commands
consistent across operating systems

2)use the Ros launch command 
	-roslaunch launchSimulation.launch
this will automaticaly start the localization and planning nodes.

A custom goal can optionally be passed to the goal:="," parameter when launching allowing the user to choose the bot's goal, if not given it will default to node 3,3.

This project is intended for educational purposes and focuses on demonstrating the core ideas of localization and global path planning rather than providing a fully optimized navigation system.

Git hub : https://github.com/SimoneMuhuho/ROS1-SOAR

Made by :

-Simone Luis Muhuho

-Barnabas Matrai