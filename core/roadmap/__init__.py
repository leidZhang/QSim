import math

from .roadmap import *


def cal_waypoint_angle(delta_x: float, delta_y: float) -> float:
    """
    Calculate the angle of the waypoint based on the delta x and delta y.

    Parameters:
    - delta_x (float): The delta x of the waypoint
    - delta_y (float): The delta y of the waypoint

    Returns:
    - float: The angle of the waypoint
    """
    if delta_x < 0 and delta_y == 0:
        return math.pi # up to bottom
    elif delta_x == 0 and delta_y > 0:
        return math.pi / 2 # right to left
    elif delta_x < 0 and delta_y > 0:
        return math.atan(delta_y / delta_x) + math.pi # right bottom
    elif delta_x > 0 and delta_y > 0:
        return math.atan(delta_y / delta_x) # left bottom
    elif delta_x > 0 and delta_y == 0:
        return 0 # bottom to up
    elif delta_x > 0 and delta_y < 0:
        return math.atan(delta_y / delta_x) # left top
    elif delta_x == 0 and delta_y < 0:
        return 3 * math.pi / 2 # left to right
    else:
        return math.atan(delta_y / delta_x) + math.pi # right top

def get_waypoint_pose(waypoint_sequence: np.ndarray, waypoint_index: int = 0) -> Tuple[list, list]:
    """
    Spawn the vehicle on the waypoints based on the waypoint index.

    Parameters:
    - waypoint_index (int): The index of the waypoint

    Returns:
    - Tuple[list, list]: The position and orientation of the vehicle on the designated waypoint
    """
    if waypoint_index < 0 or waypoint_index >= len(waypoint_sequence):
        raise ValueError('Invalid Waypoint index!')

    # handle final index
    if waypoint_index < len(waypoint_sequence) - 1:
        current_waypoint: np.ndarray = waypoint_sequence[waypoint_index]
        next_waypoint: np.ndarray = waypoint_sequence[waypoint_index+1]
    else:
        current_waypoint: np.ndarray = waypoint_sequence[waypoint_index-1]
        next_waypoint: np.ndarray = waypoint_sequence[waypoint_index]
    # calculate x, y coordinates
    x_position: float = current_waypoint[0]
    y_position: float = current_waypoint[1]
    # calcualte angle
    delta_x: float = next_waypoint[0] - current_waypoint[0]
    delta_y: float = next_waypoint[1] - current_waypoint[1]
    orientation: float = cal_waypoint_angle(delta_x, delta_y)

    return [x_position, y_position, 0], [0, 0, orientation]

def get_on_node_pose(nodes: List[float], node_index: int) -> Tuple[list, list]:
    """
    Spawn the vehicle on the nodes based on the node index.

    Parameters:
    - node_index (int): The index of the node

    Returns:
    - Tuple[list, list]: The position and orientation of the vehicle on the designated node
    """
    if node_index >= len(nodes) or node_index < 0:
        raise ValueError("Index does not exist!")
    node_pose: np.ndarray = nodes[node_index]
    x_position, y_position, orientation = node_pose
    return [x_position, y_position, 0], [0, 0, orientation]
