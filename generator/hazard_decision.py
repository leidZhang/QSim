from typing import Tuple

import numpy as np


class MapQCar:
    WIDTH: float = 0.27# 0.2
    LENGTH: float = 0.3# 0.54

    def __init__(self, use_optitrack: bool = False) -> None:
        self.correction: float = np.pi if use_optitrack else 0.0
        self.bounding_box: np.ndarray = np.array([
            np.array([-self.WIDTH/2, self.LENGTH/2]),
            np.array([self.WIDTH/2, self.LENGTH/2]),
            np.array([self.WIDTH/2, -self.LENGTH/2]),
            np.array([-self.WIDTH/2, -self.LENGTH/2])
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
    def __init__(self, actor_id: int, use_optitrack: bool = False) -> None:
        self.map_qcar: MapQCar = MapQCar(use_optitrack)
        self.actor_id: int = actor_id
        self.hazard_distance: int = 150

    def cal_waypoint_mask(
        self,
        x1: float,
        x2: float,
        y1: float,
        y2: float,
        waypoints: np.ndarray
    ) -> np.ndarray:
        return 0 < (x1 - x2) * (waypoints[:self.hazard_distance, 1] - y1) -\
            (y1 - y2) * (waypoints[:self.hazard_distance, 0] - x1)

    def get_waypoint_mask(self, waypoints: np.ndarray, hazard_state: np.ndarray, ego_state: np.ndarray) -> np.ndarray:
        # OBB detection, is the waypoint pass the object bounding box of the hazard car?
        bounding_box: np.ndarray = self.map_qcar.get_object_bounding_box_local(hazard_state, ego_state)
        bbx1, bby1 = bounding_box[0] # top right
        bbx2, bby2 = bounding_box[1] # top left
        bbx3, bby3 = bounding_box[2] # bottom left
        bbx4, bby4 = bounding_box[3] # bottom right

        mask_1: np.ndarray = self.cal_waypoint_mask(bbx1, bbx2, bby1, bby2, waypoints)
        mask_2: np.ndarray = self.cal_waypoint_mask(bbx2, bbx3, bby2, bby3, waypoints)
        mask_3: np.ndarray = self.cal_waypoint_mask(bbx3, bbx4, bby3, bby4, waypoints)
        mask_4: np.ndarray = self.cal_waypoint_mask(bbx4, bbx1, bby4, bby1, waypoints)
        mask: np.ndarray = mask_1 & mask_2 & mask_3 & mask_4
        return mask
    
    def is_in_hazard_area(self, waypoints: np.ndarray, agent_states: np.ndarray) -> bool:
        ego_state: np.ndarray = agent_states[self.actor_id]
        for i, hazard_state in enumerate(agent_states):
            if self.actor_id == i or i == 0:
                continue # we skip the ego agent

            mask = self.get_waypoint_mask(waypoints, hazard_state, ego_state)
            if np.any(mask):
                return True
        return False
