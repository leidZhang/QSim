from typing import List

import numpy as np

from .agents import CarAgent


class MapQCar:
    WIDTH: float = 0.27# 0.2
    LENGTH: float = 0.80# 0.54

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
    def __init__(self, use_optitrack: bool = False) -> None:
        self.map_qcar: MapQCar = MapQCar(use_optitrack)
        self.hazard_distance: int = 100 # waypoint numbers
        self.intersection_distance: int = 200 # waypoint numbers

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
        # OBB detection, is the waypoint pass the object bounding box of the hazard car?
        masks: List[np.ndarray] = []
        bounding_box: np.ndarray = self.map_qcar.get_object_bounding_box(hazard_state)
        for i in range(4):
            x1, y1 = bounding_box[i]
            x2, y2 = bounding_box[(i + 1) % 4]
            mask = self._cal_waypoint_mask(x1, x2, y1, y2, waypoints)
            masks.append(mask)
        return np.logical_and.reduce(masks)

    def _is_waypoint_intersected(self, ego_waypoints: np.ndarray, hazard_waypoint: np.ndarray) -> bool:
        compare_len: int = min(len(ego_waypoints), len(hazard_waypoint))
        waypoint_dists_1: np.ndarray = np.linalg.norm(ego_waypoints[-compare_len:] - hazard_waypoint[-compare_len:][::-1], axis=1)
        waypoint_dists_2: np.ndarray = np.linalg.norm(ego_waypoints[:compare_len] - hazard_waypoint[:compare_len][::-1], axis=1)
        return np.any(waypoint_dists_1 <= 0.01) or np.any(waypoint_dists_2 <= 0.01)

    def evalueate(self, ego_agent: CarAgent, hazard_agent: CarAgent) -> int:
        # get the state and waypoints of ego and hazard agent
        ego_progress: float = ego_agent.observation["progress"]
        hazard_progress: float = hazard_agent.observation["progress"]
        ego_state: np.ndarray = ego_agent.observation["state"]
        hazard_state: np.ndarray = hazard_agent.observation["state"]
        ego_waypoints: np.ndarray = ego_agent.observation["global_waypoints"]
        hazard_waypoints: np.ndarray = hazard_agent.observation["global_waypoints"]
        print(f"Agent {ego_agent.actor_id} checking the agent {hazard_agent.actor_id}")
        # If the hazard car is far away, ignore it
        if np.linalg.norm(ego_state[:2] - hazard_state[:2]) > 1.90:
            return 1
        # Check if the hazard car is in the waypoint mask
        print("Checking the waypoint mask")
        waypoint_mask: np.ndarray = self.get_waypoint_mask(ego_waypoints, hazard_state)
        if np.any(waypoint_mask):
            print(f"Agent {ego_agent.actor_id} detects {hazard_agent.actor_id} as a hazard, stop")
            return 0
        # Check if hazard waypoint is intersected with ego waypoints
        print("Checking the waypoint-obb intersection")
        if not self._is_waypoint_intersected(ego_waypoints, hazard_waypoints):
            return 1
        
        print(f"Agent {ego_agent.actor_id} detects {hazard_agent.actor_id} as a hazard")
        if ego_progress > hazard_progress:
            print(f"Agent {ego_agent.actor_id} has higher progress than {hazard_agent.actor_id}, move")
            return 1
        print(f"Agent {ego_agent.actor_id} has lower progress than {hazard_agent.actor_id}, stop")
        return 0


# TODO: Change a way to detect the intersection
# def is_traj_intersected(ego_traj: np.ndarray, hazard_traj: np.ndarray) -> bool:
#     ego_traj_len, hazard_traj_len = len(ego_traj), len(hazard_traj)
#     compare_len: int = min(ego_traj_len, hazard_traj_len)
#     # print(f"compare_len: {compare_len}")
#     # print(ego_traj, hazard_traj)

#     waypoint_dists_1: np.ndarray = np.linalg.norm(ego_traj[:compare_len] - hazard_traj[:compare_len][::-1], axis=1)
#     waypoint_dists_2: np.ndarray = np.linalg.norm(ego_traj[-compare_len:] - hazard_traj[-compare_len:], axis=1)
#     # print(f"min_dist_1", min(waypoint_dists_1), "min_dist_2", min(waypoint_dists_2))
#     mask_1: np.ndarray = waypoint_dists_1 <= 0.25
#     mask_2: np.ndarray = waypoint_dists_2 <= 0.25
#     return np.any(mask_1) or np.any(mask_2)


# def is_in_hazard_range(ego_orig: np.ndarray, agent_orig: np.ndarray, thresh: float) -> bool:
#     return np.linalg.norm(ego_orig - agent_orig) < thresh


# def has_higher_priority(ego_rank: float, agent_rank: float, ego_id: int, agent_id: int) -> bool:
#     if ego_rank > agent_rank:
#         return True
#     elif ego_id > agent_id:
#         return True
#     return False


# def make_halt_decision(
#     ego_state: np.ndarray,
#     ego_waypoints: np.ndarray,
#     ego_rank: int,
#     hazard_state: List[np.ndarray],
#     hazard_waypoints: np.ndarray,
#     hazard_rank: List[int],
#     ego_id: int,
#     hazard_id: int
# ) -> bool:
#     if not is_in_hazard_range(
#         ego_orig=ego_state[:2],
#         agent_orig=hazard_state[:2],
#         thresh=1.50
#     ):
#         return False

#     # print("Starting detect the priority")
#     if has_higher_priority(ego_rank, hazard_rank, ego_id, hazard_id):
#         # print("Ego agent has higher priority")
#         return False

#     # print("Starting detect the intersection")
#     if not is_traj_intersected(ego_waypoints, hazard_waypoints):
#         # print("No intersection detected")
#         return False

#     return True

# def get_relative_polar_coordinates(
#     ego_orig: np.ndarray,
#     agent_orig: np.ndarray
# ) -> np.ndarray:
#     relative_orig: np.ndarray = agent_orig - ego_orig
#     relative_angle: float = np.arctan2(relative_orig[1], relative_orig[0])

#     relative_angle = relative_angle # - ego_yaw # + np.pi / 2
#     relative_angle = (relative_angle + np.pi) % (2 * np.pi) - np.pi
#     relative_distance: float = np.linalg.norm(relative_orig)
#     print(f"relative_distance: {relative_distance}, relative_angle: {relative_angle / np.pi * 180}")
#     return np.array([relative_distance, relative_angle])


# class HazardDetector:
#     def __init__(
#         self,
#         dist_thresh: float = 1.20,
#         angle_thresh: float = np.pi / 6
#     ) -> None:
#         self.dist_thresh: float = dist_thresh
#         self.angle_thresh: float = angle_thresh

#     def is_hazard(self, ego_yaw: float, steering: float, agent_orig: np.ndarray) -> bool:
#         polar_coord: np.ndarray = get_relative_polar_coordinates(np.zeros(2), agent_orig)

#         print(f"polar_coord: {polar_coord[1] / np.pi * 180}")

#         dist_flag: bool = polar_coord[0] <= self.dist_thresh
#         angle_flag: bool = -self.angle_thresh + steering <= polar_coord[1] <= self.angle_thresh + steering
#         return dist_flag and angle_flag

#     def get_hazard_decision(
#         self,
#         ego_action: np.ndarray,
#         ego_state: np.ndarray,
#         agent_states: List[np.ndarray]
#     ) -> int:
#         ego_steering: float = ego_action[1]
#         ego_orig: np.ndarray = ego_state[:2]
#         ego_yaw: float = -ego_state[2]
#         rot: np.ndarray = np.array([
#             [np.cos(ego_yaw), np.sin(ego_yaw)],
#             [-np.sin(ego_yaw), np.cos(ego_yaw)]
#         ])

#         for i in range(len(agent_states)):
#             agent_orig: np.ndarray = agent_states[i][:2]
#             local_frame_orig = np.matmul(agent_orig - ego_orig, rot)
#             print(ego_state[:3], agent_states[i][:3])
#             if self.is_hazard(ego_yaw, ego_steering, local_frame_orig):

#                 return 0
#         print(ego_orig, agent_orig)
#         return 1 # no hazard detected


# class HazardDetector: # Rule is fixed, so no decision making is needed
#     def __init__(self, actor_id: int, use_optitrack: bool = False) -> None:
#         self.map_qcar: MapQCar = MapQCar(use_optitrack)
#         self.actor_id: int = actor_id
#         self.hazard_distance: int = 150

#     def cal_waypoint_mask(
#         self,
#         x1: float,
#         x2: float,
#         y1: float,
#         y2: float,
#         waypoints: np.ndarray
#     ) -> np.ndarray:
#         return 0 < (x1 - x2) * (waypoints[:self.hazard_distance, 1] - y1) -\
#             (y1 - y2) * (waypoints[:self.hazard_distance, 0] - x1)

#     def get_waypoint_mask(self, waypoints: np.ndarray, hazard_state: np.ndarray, ego_state: np.ndarray) -> np.ndarray:
#         # OBB detection, is the waypoint pass the object bounding box of the hazard car?
#         bounding_box: np.ndarray = self.map_qcar.get_object_bounding_box_local(hazard_state, ego_state)
#         bbx1, bby1 = bounding_box[0] # top right
#         bbx2, bby2 = bounding_box[1] # top left
#         bbx3, bby3 = bounding_box[2] # bottom left
#         bbx4, bby4 = bounding_box[3] # bottom right

#         mask_1: np.ndarray = self.cal_waypoint_mask(bbx1, bbx2, bby1, bby2, waypoints)
#         mask_2: np.ndarray = self.cal_waypoint_mask(bbx2, bbx3, bby2, bby3, waypoints)
#         mask_3: np.ndarray = self.cal_waypoint_mask(bbx3, bbx4, bby3, bby4, waypoints)
#         mask_4: np.ndarray = self.cal_waypoint_mask(bbx4, bbx1, bby4, bby1, waypoints)
#         mask: np.ndarray = mask_1 & mask_2 & mask_3 & mask_4
#         return mask

#     def is_in_hazard_area(self, waypoints: np.ndarray, agent_states: np.ndarray) -> bool:
#         ego_state: np.ndarray = agent_states[self.actor_id]
#         for i, hazard_state in enumerate(agent_states):
#             if self.actor_id == i or i == 0:
#                 continue # we skip the ego agent

#             mask = self.get_waypoint_mask(waypoints, hazard_state, ego_state)
#             if np.any(mask):
#                 return True
#         return False
