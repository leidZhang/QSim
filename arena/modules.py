from queue import Queue
from copy import deepcopy
from threading import Event
from typing import List, Tuple

import numpy as np

from core.roadmap import ACCRoadMap
from core.environment.simulator import QLabSimulator
from system.settings import EGO_VEHICLE_TASK
from system.settings import COLLISION_PENALTY
from .detectors import is_collided

MAX_LOOKAHEAD_INDICES: int = 200


class WaypointProcessor:
    def __init__(self, roadmap: ACCRoadMap, task: List[int]) -> None:
        self.waypoints: np.ndarray = roadmap.generate_path(task)
        self.original: np.ndarray = deepcopy(self.waypoints)
        self.local_waypoints: np.ndarray = np.zeros((MAX_LOOKAHEAD_INDICES, 2))

    def setup(self, init_waypoint_index: int, ego_state: np.ndarray) -> None:
        orig, _, rot = self.cal_vehicle_state(ego_state)
        self.current_waypoint_index: int = init_waypoint_index
        self.next_waypoints: np.ndarray = self.waypoints[self.current_waypoint_index:]
        self.handle_observation(orig, rot)
    
    def cal_vehicle_state(self, ego_state: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        orig: np.ndarray = ego_state[:2]
        yaw: float = -ego_state[2] # + np.pi
        rot: np.ndarray = np.array([
            [np.cos(yaw), np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)]
        ])
        return orig, yaw, rot
    
    def cal_local_waypoints(self, orig: np.ndarray, rot: np.ndarray) -> np.ndarray:
        # get the local waypoints
        local_waypoints: np.ndarray = np.roll(
            self.waypoints, -self.current_waypoint_index, axis=0
        )[:MAX_LOOKAHEAD_INDICES]
        # get the distance to the waypoints
        self.norm_dist: np.ndarray = np.linalg.norm(local_waypoints - orig, axis=1)
        # get the index of the closest waypoint
        self.dist_ix: int = np.argmin(self.norm_dist)
        # update the current waypoint index
        self.current_waypoint_index = (self.current_waypoint_index + self.dist_ix) % self.waypoints.shape[0]
        # clear pasted waypoints
        self.next_waypoints = self.next_waypoints[self.dist_ix:] 
    
    def handle_observation(self, orig: np.ndarray, rot: np.ndarray) -> None:
        # in case of less than 200 waypoints
        if self.next_waypoints.shape[0] < MAX_LOOKAHEAD_INDICES:
            slop = MAX_LOOKAHEAD_INDICES - self.next_waypoints.shape[0]
            self.next_waypoints = np.concatenate([self.next_waypoints, self.waypoints[:slop]])
        # add waypoints info to observation
        self.local_waypoints = np.matmul(self.next_waypoints[:MAX_LOOKAHEAD_INDICES] - orig, rot)

    def check_next_task(self) -> bool:
        if self.current_waypoint_index >= len(self.original) - 250:
            self.next_waypoints = np.concatenate([self.waypoints, self.original])
            self.waypoints = self.next_waypoints                  

    def execute(self, ego_state: np.ndarray) -> None:
        orig, _, rot = self.cal_vehicle_state(ego_state)
        self.cal_local_waypoints(orig, rot)
        self.handle_observation(orig, rot)    
        self.check_next_task()

    def get_deviation(self) -> float:
        return self.norm_dist[self.dist_ix]
    
    def is_completed(self) -> bool:
        return self.current_waypoint_index >= len(self.original)


# TODO: Implement the real world environment later
class RealWorldEnv: 
    def __init__(
        self, 
        roadmap: ACCRoadMap, 
        sim: QLabSimulator, 
        start_poses: List[Tuple[List[float], List[float]]], 
        event: Event
    ) -> None:
        self.sim: QLabSimulator = sim
        self.event: Event = event
        self.start_poses: List[List[int]] = start_poses        
        self.sim.render_map() # render the map
        self.__generate_cars() # generate the cars
        self.waypoint_processor: WaypointProcessor = WaypointProcessor(roadmap, EGO_VEHICLE_TASK)

    def __generate_cars(self) -> None:
        for i in range(1, len(self.start_poses)): # reset the bot cars
            self.sim.add_car(self.start_poses[i][0], self.start_poses[i][1])    

    def reset_ego_vehicle(self) -> None:
        self.sim.reset_map(self.start_poses[0][0], self.start_poses[0][1])             

    # TODO: change this method to reset the real world environment
    def reset(self, state_queue: Queue) -> Tuple[bool, float, np.ndarray]:
        print("Resetting the environment...")
        # reset the ego vehicle position
        self.reset_ego_vehicle()
        # reset the waypoint processor                
        ego_states: List[np.ndarray] = state_queue.get()
        self.waypoint_processor.setup(0, ego_states[0])
        self.last_waypoint_index: int = 0
        return False, 0, np.zeros(2) # done, rewards, actions
    
    def handle_reward(
        self, 
        action: np.ndarray, 
        poses: List[np.ndarray], 
    ) -> Tuple[bool, float]:
        # calculate the collision penalty        
        for bot_poses in poses[1:]:
            orig_1: np.ndarray = poses[0][:2]
            yaw_1: float = -poses[0][2]
            orig_2: np.ndarray = bot_poses[:2]
            yaw_2: float = -bot_poses[2]
            if is_collided(orig_1, yaw_1, orig_2, yaw_2):
                self.event.set() # set the early stop event
                return True, COLLISION_PENALTY        

        # calculate the progress reward
        current_waypoint_index: int = self.waypoint_processor.current_waypoint_index
        reward = current_waypoint_index - self.last_waypoint_index
        # calculate the deviation penalty
        deviation: float = self.waypoint_processor.get_deviation()
        reward -= -max(0.0, 1.3 * (current_waypoint_index - self.last_waypoint_index) * (deviation - 0.031))
        # calculate the speed reward
        reward += 4 * (action[0] - 0.04)

        # update the last waypoint index
        self.last_waypoint_index = current_waypoint_index

        return self.waypoint_processor.is_completed(), reward
    
    def step(self, action: np.ndarray, state_queue: Queue) -> Tuple[bool, float, np.ndarray]:
        poses: List[np.ndarray] = state_queue.get() # 0: ego vehicle, 1-2: bot car
        self.waypoint_processor.execute(poses[0]) # update the ego vehicle state info
        done, reward = self.handle_reward(action, poses)
        return done, reward, action
