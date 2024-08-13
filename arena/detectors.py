from typing import List

import numpy as np


class EnvQCarRef:
    def __init__(self, threshold: float = 0.2) -> None:
        self.width: float = 0.40# 0.22
        self.length: float = 0.22# 0.40
        self.corners = np.array([
            [self.width/2 + threshold, self.length/2],
            [-self.width/2 + threshold, self.length/2],
            [-self.width/2, -self.length/2],
            [self.width/2, -self.length/2]
        ])

    def get_corners(self, orig: np.ndarray, yaw: float) -> np.ndarray:
        rot = np.array([
            [np.cos(yaw), np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)]
        ])
        return orig + np.dot(self.corners, rot.T)
    

def project_polygon(corner: np.ndarray, axis: List[float]) -> List[float]:
    return [np.dot(corner, axis) for corner in corner]


def overlap(proj_1: List[float], proj_2: List[float]) -> bool:
    return max(proj_1) >= min(proj_2) and max(proj_2) > min(proj_1)


def get_axes(corners: np.ndarray) -> List[float]:
    axes: List[float] = []
    for i in range(len(corners)):
        p1: float = corners[i]
        p2: float = corners[(i + 1) % len(corners)]
        edge: float = p2 - p1
        normal = np.array([-edge[1], edge[0]])  # Perpendicular vector
        axes.append(normal / np.linalg.norm(normal))  # Normalize the axis
    return axes


def separating_axis_therem(corners_1: np.ndarray, corners_2: np.ndarray) -> bool:
    axes_1: List[float] = get_axes(corners_1)
    axes_2: List[float] = get_axes(corners_2)
    for axis in axes_1 + axes_2:
        proj_1: List[float] = project_polygon(corners_1, axis)
        proj_2: List[float] = project_polygon(corners_2, axis)
        if not overlap(proj_1, proj_2):
            return False
    return True


def is_collided(orig_1, yaw_1, orig_2, yaw_2, actor_type: EnvQCarRef):
    corners_1 = actor_type.get_corners(orig_1, yaw_1)
    corners_2 = actor_type.get_corners(orig_2, yaw_2)
    return separating_axis_therem(corners_1, corners_2)
    

if __name__ == "__main__":
    orig_1 = np.array([0, 0])
    yaw_1 = np.pi/180 * 250
    orig_2 = np.array([0.3, 0])
    yaw_2 = np.pi/4

    collision = is_collided(orig_1, yaw_1, orig_2, yaw_2)
    print("Collision detected:", collision)
