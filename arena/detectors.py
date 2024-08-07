import numpy as np


class EnvQCarRef:
    def __init__(self, buffer: float = 0.20) -> None:
        self.width: float = 0.21 + buffer
        self.length: float = 0.39 + buffer
        self.corners = np.array([
            [self.width/2, self.length/2],
            [-self.width/2, self.length/2],
            [-self.width/2, -self.length/2],
            [self.width/2, -self.length/2]
        ])

    def get_corners(self, orig: np.ndarray, yaw: float) -> np.ndarray:
        rot = np.array([
            [np.cos(yaw), np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)]
        ])
        return orig + np.dot(self.corners, rot.T)
    

def separating_axis_theorem(corners1, corners2) -> bool:
    axes: np.ndarray = np.vstack([corners1, corners2])
    axes = np.diff(axes, axis=0)
    axes = np.append(axes, [axes[-1]], axis=0)
    axes = np.cross(axes, [0, 0, 1])[:, :2]

    for axis in axes:
        projection1: np.ndarray = np.dot(corners1, axis)
        projection2: np.ndarray = np.dot(corners2, axis)
        if max(projection1) < min(projection2) or max(projection2) < min(projection1):
            return False
        
    return True


def is_collided(orig_1, yaw_1, orig_2, yaw_2):
    corners_1: np.ndarray = EnvQCarRef().get_corners(orig_1, yaw_1)
    corners_2: np.ndarray = EnvQCarRef().get_corners(orig_2, yaw_2)
    return separating_axis_theorem(corners_1, corners_2)
