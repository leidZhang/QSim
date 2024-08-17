from typing import List
from shapely import Polygon

import numpy as np

REAR_LENGTH: float = 0.075
FORWARD_LENGTH: float = 0.425
HALF_WIDTH: float = 0.20 / 2


class EnvQCarRef:
    def __init__(self) -> None:
        self.box = np.array([
            [-HALF_WIDTH, -REAR_LENGTH], 
            [HALF_WIDTH, -REAR_LENGTH],
            [HALF_WIDTH, FORWARD_LENGTH],
            [-HALF_WIDTH, FORWARD_LENGTH]
        ])

    def get_poly(self, state: np.ndarray) -> Polygon:
        x, y, yaw = state[:3]
        rot = np.array([
            [-np.sin(yaw), np.cos(yaw)],
            [np.cos(yaw), np.sin(yaw)]
        ])
        corrected_box: np.ndarray = np.dot(self.box, rot) + np.array([x, y])
        return Polygon(corrected_box)
    
def is_collided(
    state_1: np.ndarray, 
    state_2: np.ndarray, 
    actor_type: EnvQCarRef
) -> bool:
    poly_1: Polygon = actor_type.get_poly(state_1)
    poly_2: Polygon = actor_type.get_poly(state_2)
    if poly_1.intersects(poly_2):
        intersection_area: float = poly_1.intersection(poly_2).area
        return intersection_area > 0.0004
    return False

# class EnvQCarRef:
#     def __init__(self, threshold: float = 0.10) -> None:
#         self.width: float = 0.40# 0.22
#         self.length: float = 0.22# 0.40
#         self.corners = np.array([
#             [self.width/2, self.length/2],
#             [-self.width/2, self.length/2],
#             [-self.width/2, -self.length/2],
#             [self.width/2, -self.length/2]
#         ])

#     def get_corners(self, orig: np.ndarray, yaw: float) -> np.ndarray:
#         rot: np.ndarray = np.array([
#             [np.cos(yaw), np.sin(yaw)],
#             [-np.sin(yaw), np.cos(yaw)]
#         ])

#         return orig + np.dot(self.corners, rot.T)

# def project_polygon(corners: np.ndarray, axis: List[float]) -> List[float]:
#     return [np.dot(corner, axis) for corner in corners]


# def overlap(proj_1: List[float], proj_2: List[float]) -> bool:
#     return max(proj_1) >= min(proj_2) and max(proj_2) > min(proj_1)


# def get_axes(corners: np.ndarray) -> List[float]:
#     axes: List[float] = []
#     for i in range(len(corners)):
#         p1: float = corners[i]
#         p2: float = corners[(i + 1) % len(corners)]
#         edge: float = p2 - p1
#         normal = np.array([-edge[1], edge[0]])  # Perpendicular vector
#         axes.append(normal / np.linalg.norm(normal))  # Normalize the axis
#     return axes


# def separating_axis_therem(corners_1: np.ndarray, corners_2: np.ndarray) -> bool:
#     axes_1: List[float] = get_axes(corners_1)
#     axes_2: List[float] = get_axes(corners_2)
#     for axis in axes_1 + axes_2:
#         proj_1: List[float] = project_polygon(corners_1, axis)
#         proj_2: List[float] = project_polygon(corners_2, axis)
#         if not overlap(proj_1, proj_2):
#             return False
#     return True


# def is_collided(state_1: np.ndarray, state_2: np.ndarray, actor_type: EnvQCarRef):
#     corners_1 = actor_type.get_corners(state_1[:2], state_1[2], state_1[3])
#     corners_2 = actor_type.get_corners(state_2[:2], state_2[2], state_2[3])
#     return separating_axis_therem(corners_1, corners_2)
    
