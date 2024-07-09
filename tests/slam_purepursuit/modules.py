import os
import time
from typing import Dict
from queue import Queue
from typing import Tuple
from multiprocessing import Queue as MPQueue

import numpy as np

from core.utils.ipc_utils import fetch_latest_in_queue
from core.utils.ipc_utils import put_latest_in_queue
from core.utils.performance import elapsed_time
from core.policies.pure_persuit import PurePursuiteAdaptor
from core.qcar.vehicle import PhysicalCar
from tests.resnet.model import ObstacleDetection
from tests.resnet.detector import ResNetDetector
from tests.traditional.exceptions import HaltException
# from constants import MAX_LOOKAHEAD_INDICES

MAX_LOOKAHEAD_INDICES = 200
CALIBRATION_POSE = [0,2,-np.pi/2]


class ResNetWrapper:
    def __init__(self) -> None:
        project_path: str = os.getcwd()
        model: ObstacleDetection = ObstacleDetection()
        weights_path: str = os.path.join(project_path, "tests/resnet/obs_model.pth")
        self.detector: ResNetDetector = ResNetDetector(model=model, weights_path=weights_path)

    def execute(self, data_queue: MPQueue, response_queue: MPQueue) -> None:
        image: np.ndarray = fetch_latest_in_queue(data_queue)
        if image is not None:
            cls_pred, dis_pred = self.detector(image)
            put_latest_in_queue({'cls_pred': cls_pred, 'dis_pred': dis_pred}, response_queue)


class WaypointProcessor:
    def __init__(self, waypoints: np.ndarray, max_lookahead_distance: float = 0.5) -> None:
        self.waypoints: np.ndarray = waypoints
        print(f"Task length: {len(self.waypoints)}")
        self.local_waypoints: np.ndarray = np.zeros((MAX_LOOKAHEAD_INDICES, 2))
        self.policy: PurePursuiteAdaptor = PurePursuiteAdaptor(
            max_lookahead_distance=max_lookahead_distance
        )

    def setup(self, init_waypoint_index: int, data_queue: Queue) -> None:
        ego_state: np.ndarray = self.get_ego_state(data_queue)
        orig, _, rot = self.cal_vehicle_state(ego_state)
        self.current_waypoint_index: int = init_waypoint_index
        self.next_waypoints: np.ndarray = self.waypoints[self.current_waypoint_index:]
        self.handle_observation(orig, rot)
        
    def get_ego_state(self, data_queue: Queue) -> np.ndarray:
        if data_queue.empty():
            return None
        return data_queue.get()
    
    def cal_vehicle_state(
        self, 
        ego_state: np.ndarray
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        orig: np.ndarray = ego_state[:2]
        # self.history.append(orig)
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

    def execute(self, data_queue: MPQueue, response_queue: MPQueue) -> None:
        ego_state = self.get_ego_state(data_queue) 
        if ego_state is None:
            return
        
        orig, yaw, rot = self.cal_vehicle_state(ego_state)
        self.cal_local_waypoints(orig, rot)
        self.handle_observation(orig, rot)
        put_latest_in_queue(self.local_waypoints, response_queue)
        # for debug purpose
        # print(f"Received ego state: {ego_state}")
        # print(f"Current index is {self.current_waypoint_index}")
        # print(f"Current Position: {orig}, Orientation: {yaw}")
        # print(f"Waypoint position {self.waypoints[self.current_waypoint_index]}")


class PurepursuitCar(PhysicalCar):
    def __init__(
        self, 
        throttle_coeff: float = 0.3, 
        steering_coeff: float = 0.5,
        desired_speed: float = 0.08
    ) -> None:
        super().__init__(throttle_coeff, steering_coeff)
        self.desired_speed: float = desired_speed
        self.brake_time: float = (desired_speed * 10 - 1.0) / 1.60
        self.observation: dict = {}
        self.policy: PurePursuiteAdaptor = PurePursuiteAdaptor(max_lookahead_distance=0.5)

    def setup(self) -> None: # for throttle pid control
        pass

    def handle_detection(self, observe_queue: dict) -> None:
        stop_flags: Dict[str, bool] = fetch_latest_in_queue(observe_queue)

        if stop_flags is None: return
        # print(stop_flags) # received flags
        # stop if detects the red light
        if stop_flags['red_light']:
            halt_time: float = 0.1
            raise HaltException(stop_time=halt_time)
        # stop if detects the stop sign
        if stop_flags['stop_sign']:
            halt_time: float = 3 + self.brake_time
            raise HaltException(stop_time=halt_time)
        
    def handle_evade(self, obstacle_queue: MPQueue) -> np.ndarray:
        result: tuple = fetch_latest_in_queue(obstacle_queue)
        if result is not None:
            print(result)
        
    def handle_control(self, control_queue: MPQueue, evade_traj: np.ndarray = None) -> None:
        local_waypoints: np.ndarray = fetch_latest_in_queue(control_queue)
        if local_waypoints is not None:
            self.observation['waypoints'] = local_waypoints
            # local_waypoints += evade_traj
            self.action, _ = self.policy.execute(obs=self.observation)
            self.running_gear.read_write_std(self.desired_speed, self.action[1] * 0.5)

    def execute(self, observe_queue: MPQueue, control_queue: MPQueue, obstacle_queue: MPQueue) -> None:
        try:
            start: float = time.time()
            self.handle_detection(observe_queue)
            self.handle_evade(obstacle_queue)
            self.handle_control(control_queue)
            duration: float = elapsed_time(start)
            time.sleep(max(0.002 - duration, 0)) 
        except HaltException as e:
            print(f"Stopping the car for {e.stop_time:.2f} seconds")
            self.halt_car(steering=self.action[1], halt_time=e.stop_time)
