import numpy as np

from core.roadmap.roadmap import ACCRoadMap

#[10, 4, 14, 20, 22, 10]
#[10, 4, 6, 8]
def prepare_task(node_sequence: list = [10, 2, 4, 14, 20, 22]*5) -> None:
    roadmap: ACCRoadMap = ACCRoadMap()
    return roadmap.generate_path(node_sequence=node_sequence)

def correct_traj(waypoints: np.ndarray) -> np.ndarray:
    shift: np.ndarray = np.array([0.15, 0.11])
    yaw_diff: float = -2.5 * np.pi / 180.0
    rot = np.array([
        [np.cos(yaw_diff), -np.sin(yaw_diff)],
        [np.sin(yaw_diff), np.cos(yaw_diff)]
    ])
    waypoints -= shift
    waypoints = np.matmul(rot, waypoints.T).T
    #waypoints += np.matmul(rot, shift)
    return waypoints