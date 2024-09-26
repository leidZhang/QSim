from typing import List, Dict

import numpy as np

from core.roadmap.raster_map import to_pixel
from settings import REFERENCE_POSE


def is_in_area_aabb(state: np.ndarray, area_box: Dict[str, float]) -> bool:
    return area_box["min_x"] <= state[0] <= area_box["max_x"] and area_box["min_y"] <= state[1] <= area_box["max_y"]


class MapQCar:
    WIDTH: float = 0.215 # 0.2
    LENGTH: float = 0.54 # 0.54

    def __init__(self, use_optitrack: bool = False) -> None:
        self.correction: float = np.pi if use_optitrack else 0.0
        self.bounding_box: np.ndarray = np.array([
            np.array([-self.WIDTH / 2, self.LENGTH / 2]),
            np.array([self.WIDTH / 2, self.LENGTH / 2]),
            np.array([self.WIDTH / 2, -self.LENGTH / 2]),
            np.array([-self.WIDTH / 2, -self.LENGTH / 2])
        ])

    def get_object_bounding_box(self, state: np.ndarray) -> np.ndarray:
        orig, yaw = state[:2], state[2]
        yaw: float = -state[2] + self.correction
        rot: np.ndarray = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]
        ])
        return np.matmul(rot, self.bounding_box.T).T + orig

    def get_object_bounding_box_local(self, agent_state: np.ndarray, ego_state: np.ndarray) -> np.ndarray:
        agent_box: np.ndarray = self.get_object_bounding_box(agent_state)
        orig, yaw = ego_state[:2], ego_state[2]
        rot: np.ndarray = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)],
        ])
        return np.matmul(agent_box - orig, rot)


class HazardDetector: # Rule is fixed, so no decision making is needed
    def __init__(self, use_optitrack: bool = False) -> None:
        self.map_qcar: MapQCar = MapQCar(use_optitrack)
        self.hazard_distance: int = 80 # waypoint numbers
        # self.intersection_distance: int = 200 # waypoint numbers

    def _cal_waypoint_mask(
        self,
        x1: float,
        x2: float,
        y1: float,
        y2: float,
        waypoints: np.ndarray
    ) -> np.ndarray:
        return 0 < (x1 - x2) * (waypoints[:self.hazard_distance, 1] - y1) -\
            (y1 - y2) * (waypoints[:self.hazard_distance, 0] - x1)

    def get_waypoint_mask(self, waypoints: np.ndarray, hazard_state: np.ndarray) -> np.ndarray:
        # OBB detection, does the waypoint pass the object bounding box of the hazard car?
        masks: List[np.ndarray] = []
        bounding_box: np.ndarray = self.map_qcar.get_object_bounding_box(hazard_state)
        for i in range(4):
            x1, y1 = bounding_box[i]
            x2, y2 = bounding_box[(i + 1) % 4]
            mask = self._cal_waypoint_mask(x1, x2, y1, y2, waypoints)
            masks.append(mask)
        return np.logical_and.reduce(masks)

    def _is_waypoint_intersected(self, ego_waypoints: np.ndarray, hazard_waypoint: np.ndarray) -> bool:
        ego_polyline: np.ndarray = to_pixel(ego_waypoints, REFERENCE_POSE[:3])
        hazard_polyline: np.ndarray = to_pixel(hazard_waypoint, REFERENCE_POSE[:3])
        set1: set = set(map(tuple, ego_polyline))
        set2: set = set(map(tuple, hazard_polyline))
        common_points: set = set1.intersection(set2)
        return len(common_points) != 0

    def evalueate(self, ego_traj: np.ndarray, hazard_traj: np.ndarray) -> int:
        if not self._is_waypoint_intersected(ego_traj, hazard_traj):
            return 1
        return 0
