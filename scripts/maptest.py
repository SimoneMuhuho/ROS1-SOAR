#!/usr/bin/env python3
import rospy
from nav_msgs.srv import GetMap

rospy.init_node("debug_mapinfo")
rospy.wait_for_service('static_map')
from nav_msgs.srv import GetMap
get_map = rospy.ServiceProxy('static_map', GetMap)
m = get_map().map

print("=== MAP INFO ===")
print(f"width x height: {m.info.width} x {m.info.height}")
print(f"resolution: {m.info.resolution} m/pixel")
print(f"origin: ({m.info.origin.position.x:.3f}, {m.info.origin.position.y:.3f})")