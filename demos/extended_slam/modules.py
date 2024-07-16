import time
from typing import List, Tuple
from threading import Event

import numpy as np

from core.qcar.vehicle import PhysicalCar
from core.utils.ipc_utils import EventQueue
from core.utils.performance import skip
from core.policies.pure_persuit import PurePursuiteAdaptor
from .filters import RollingAverageFilter

MAX_LOOKAHEAD_INDICES = 200


class WaypointProcessor:
    def __init__(self, waypoints: np.ndarray, max_lookahead_distance: float = 0.5) -> None:
        self.waypoints: np.ndarray = waypoints
        print(f"Task length: {len(self.waypoints)}")
        self.local_waypoints: np.ndarray = np.zeros((MAX_LOOKAHEAD_INDICES, 2))
        self.policy: PurePursuiteAdaptor = PurePursuiteAdaptor(
            max_lookahead_distance=max_lookahead_distance
        )

    def setup(self, init_waypoint_index: int, data_queue: EventQueue) -> None:
        ego_state: np.ndarray = self.get_ego_state(data_queue)
        orig, _, rot = self.cal_vehicle_state(ego_state)
        self.current_waypoint_index: int = init_waypoint_index
        self.next_waypoints: np.ndarray = self.waypoints[self.current_waypoint_index:]
        self.handle_observation(orig, rot)
        
    def get_ego_state(self, data_queue: EventQueue) -> np.ndarray:
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

    def execute(self, data_queue: EventQueue, response_queue: EventQueue) -> None:
        ego_state = self.get_ego_state(data_queue) 
        if ego_state is None:
            return
        
        orig, yaw, rot = self.cal_vehicle_state(ego_state)
        self.cal_local_waypoints(orig, rot)
        self.handle_observation(orig, rot)
        response_queue.put(self.local_waypoints.copy())
        # for debug purpose
        # print(f"Received ego state: {ego_state}")
        # print(f"Current index is {self.current_waypoint_index}")
        # print(f"Current Position: {orig}, Orientation: {yaw}")
        # print(f"Waypoint position {self.waypoints[self.current_waypoint_index]}")


class SLAMCar(PhysicalCar):
    def __init__(
        self, 
        events: List[Event],
        desired_speed: float = 0.8,
        max_lookahead_distance: float = 0.5
    ) -> None:
        super().__init__(0.08, 0.5)
        # observation attributes
        self.observation: dict = {}    
        # detection attributes
        self.events: List[Event] = events
        self.last_stop_sign: float = 0
        self.brake_time: float = max((desired_speed * 10 - 1.0) / 1.60, 0)        
        self.traffic_filters: List[RollingAverageFilter] = [
            RollingAverageFilter(events[1], 0.5, 3), # for stop sign detection
            RollingAverageFilter(events[2], 0.3, 10) # for traffic light detection
        ]
        # control attributes    
        self.action: np.ndarray = np.array([0.0, 0.0])
        self.desired_speed: float = desired_speed
        self.policy: PurePursuiteAdaptor = PurePursuiteAdaptor(max_lookahead_distance)

    def terminate(self) -> None:
        self.running_gear.terminate()

    def handle_detection(self, detection_queue: EventQueue) -> None:
        res: int = detection_queue.get()
        if res is not None:
            self.traffic_filters[0](res, time.time() - self.last_stop_sign >= 5)
            self.traffic_filters[1](res)
        
    def handle_stop_sign(self) -> None:
        self.halt_car(self.action[1], self.brake_time)
        time.sleep(3.0) # stop for 3 seconds
        self.last_stop_sign = time.time()
        self.events[1].clear()    

    def handle_control(self, control_queue: EventQueue) -> None:
        local_waypoints: np.ndarray = control_queue.get()
        if local_waypoints is not None:
            self.observation['waypoints'] = local_waypoints
            self.action, _ = self.policy.execute(obs=self.observation)
            throttle: float = self.throttle_coeff * self.action[0]
            steering: float = self.steering_coeff * self.action[1]
            self.running_gear.read_write_std(throttle, steering, self.leds)