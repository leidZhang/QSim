from typing import Dict, Set, Any

import numpy as np

# scale for acc competition
ACC_SCALE = 0.002
# origin map offset
X_OFFSET = 1134
Y_OFFSET = 2363
# map road radiuses
INNER_LANE_RADIUS = 305.5
OUTER_LANE_RADIUS = 438
TRAFFIC_CIRCLE_RADIUS = 333
ONE_WAY_STREET_RADIUS = 350
KINK_STREET_RADIUS = 375
# pie
PI = np.pi
HALF_PI = PI / 2
# node positions
NODE_POSES_RIGHT_COMMON = [
    [1134, 2299, -HALF_PI], # 0
    [1266, 2323, HALF_PI], # 1
    [1688, 2896, 0], # 2
    [1688, 2763, PI], # 3
    [2242, 2323, HALF_PI], # 4
    [2109, 2323, -HALF_PI], # 5
    [1632, 1822, PI], # 6
    [1741, 1955, 0], # 7
    [766, 1822, PI], # 8
    [766, 1955, 0], # 9
    [504, 2589, -42*PI/180], # 10
]

NODE_POSES_RIGHT_LARGE_MAP = [
    [1134, 1300, -HALF_PI], # 11
    [1134, 1454, -HALF_PI], # 12
    [1266, 1454, HALF_PI], # 13
    [2242, 905, HALF_PI], # 14
    [2109, 1454,-HALF_PI], # 15
    [1580, 540, -80.6*PI/180], # 16
    [1854.4, 814.5, -9.4*PI/180], # 17
    [1440, 856, -138*PI/180], # 18
    [1523, 958, 42*PI/180], # 19
    [1134, 153, PI], # 20
    [1134, 286, 0], # 21
    [159, 905, -HALF_PI], # 22
    [291, 905, HALF_PI], # 23
    [531.5, 2778, -42*PI/180] # 24
]

# edge positions
EDGE_CONFIGS_RIGHT_COMMON = [
    [0, 2, OUTER_LANE_RADIUS],
    [1, 7, INNER_LANE_RADIUS],
    [1, 8, OUTER_LANE_RADIUS],
    [2, 4, OUTER_LANE_RADIUS],
    [3, 1, INNER_LANE_RADIUS],
    [4, 6, OUTER_LANE_RADIUS],
    [5, 3, INNER_LANE_RADIUS],
    [6, 0, OUTER_LANE_RADIUS],
    [6, 8, 0],
    [7, 5, INNER_LANE_RADIUS],
    [8, 10, ONE_WAY_STREET_RADIUS],
    [9, 0, INNER_LANE_RADIUS],
    [9, 7, 0],
    [10, 1, INNER_LANE_RADIUS],
    [10, 2, INNER_LANE_RADIUS],
]

EDGE_CONFIGS_RIGHT_LARGE_MAP = [
    [1, 13, 0],
    [4, 14, 0],
    [6, 13, INNER_LANE_RADIUS],
    [7, 14, OUTER_LANE_RADIUS],
    [8, 23, INNER_LANE_RADIUS],
    [9, 13, OUTER_LANE_RADIUS],
    [11, 12, 0],
    [12, 0, 0],
    [12, 7, OUTER_LANE_RADIUS],
    [12, 8, INNER_LANE_RADIUS],
    [13, 19, INNER_LANE_RADIUS],
    [14, 16, TRAFFIC_CIRCLE_RADIUS],
    [14, 20, TRAFFIC_CIRCLE_RADIUS],
    [15, 5, OUTER_LANE_RADIUS],
    [15, 6, INNER_LANE_RADIUS],
    [16, 17, TRAFFIC_CIRCLE_RADIUS],
    [16, 18, INNER_LANE_RADIUS],
    [17, 15, INNER_LANE_RADIUS],
    [17, 16, TRAFFIC_CIRCLE_RADIUS],
    [17, 20, TRAFFIC_CIRCLE_RADIUS],
    [18, 11, KINK_STREET_RADIUS],
    [19, 17, INNER_LANE_RADIUS],
    [20, 22, OUTER_LANE_RADIUS],
    [21, 16, INNER_LANE_RADIUS],
    [22, 9, OUTER_LANE_RADIUS],
    [22, 10, OUTER_LANE_RADIUS],
    [23, 21, INNER_LANE_RADIUS],
]

# waypoints
WAYPOINT_ROTATION = [0, 0, 0]
WAYPOINT_SCALE = [0.01, 0.01, 0.02]

# actor offset
ACC_X_OFFSET = 0.13
ACC_Y_OFFSET = 1.67

# ajacency list for task dispatching
ACC_GRAPH_RIGHT: Dict[int, Set[int]] = {
    0: {2},
    1: {7, 8, 13},
    2: {4},
    3: {1},
    4: {6, 14},
    5: {3},
    6: {0, 8, 13},
    7: {5, 14},
    8: {10, 23},
    9: {0, 7, 13},
    10: {2, 1},
    11: {12},
    12: {0, 7, 8},
    13: {19},
    14: {16, 20},
    15: {5, 6},
    16: {17, 18},
    17: {15, 16},
    18: {11},
    19: {17},
    20: {22},
    21: {16},
    22: {9, 10},
    23: {21},
}

PIXEL_PER_METER = 192 * 0.48
TRAFFIC_LIGHT_SIGNALS = ["green", "yellow", "red"]
TRAFFIC_LIGHT_COLORS = [
    (0, 255, 0),
    (0, 181, 247),
    (0, 0, 255)
]
ACC_MAP_PARAMS = {
    "lanes":            ((255, 255, 255), 1),
    "green_lights":     ((0, 255, 0), 2),
    "yellow_lights":    ((0, 181, 247), 3),
    "red_lights":       ((0, 0, 255), 4),
    "objects":          ((255, 0, 255), 5),
    "ego":              ((255, 0, 0), 6),
}