from typing import Tuple, Any
import time

import cv2
import numpy as np

from qvl.qlabs import QuanserInteractiveLabs
from pal.utilities.math import Calculus
from core.qcar import VirtualCar, VirtualCSICamera
from core.qcar.constants import WHEEL_RADIUS, ENCODER_COUNTS_PER_REV, PIN_TO_SPUR_RATIO
from core.utils.performance import elapsed_time
from system.settings import MAX_LOOKAHEAD_INDICES, RESOLUTION
from .policy import ReinformerPolicy


class WaypointCar(VirtualCar):
    def __init__(
            self,
            actor_id: int,
            dt: float,
            qlabs: QuanserInteractiveLabs,
            throttle_coeff: float = 0.3,
            steering_coeff: float = 0.5
        ) -> None:
        self.diff = Calculus().differentiator_variable(dt)
        _ = next(self.diff)
        super().__init__(actor_id, dt, qlabs, throttle_coeff, steering_coeff)
        self.front_csi: VirtualCSICamera = VirtualCSICamera(id=3)
        self.observation: dict = {}

    def handle_observation(self, orig: np.ndarray, rot: np.ndarray, image) -> None:
        # in case of less than 200 waypoints
        if self.next_waypoints.shape[0] < MAX_LOOKAHEAD_INDICES:
            slop = MAX_LOOKAHEAD_INDICES - self.next_waypoints.shape[0]
            self.next_waypoints = np.concatenate([self.next_waypoints, self.waypoints[:slop]])
        # add waypoints info to observation
        # print(f'before convert of close waypoint: {self.next_waypoints[0]}')
        self.observation['waypoints'] = np.matmul(self.next_waypoints[:MAX_LOOKAHEAD_INDICES] - orig, rot)
        self.observation['image'] = cv2.resize(image, RESOLUTION).astype(np.float32) / 255.0

    def setup(self, waypoints: np.ndarray, init_waypoint_index: int = 0) -> None:
        image: np.ndarray = self.front_csi.await_image()  # init image till image is not none
        self.waypoints: np.ndarray = waypoints
        self.ego_state: np.ndarray = self.get_ego_state()
        orig, _, rot = self.cal_vehicle_state(self.ego_state)
        self.current_waypoint_index: int = init_waypoint_index
        self.next_waypoints: np.ndarray = self.waypoints[self.current_waypoint_index:]
        self.handle_observation(orig, rot, image)

    def update_state(self, image) -> None:
        # get the ego state from the qlabs
        self.ego_state = self.get_ego_state() 
        # get the original and rotation matrix
        orig, _, rot = self.cal_vehicle_state(self.ego_state)
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
        # add waypoint to the observation
        self.handle_observation(orig, rot, image)

    def estimate_speed(self) -> float:
        encoder_counts: np.ndarray = self.running_gear.motorEncoder
        encoder_speed: float = self.diff.send((encoder_counts[0], self.monitor.dt))
        return encoder_speed * (1 / (ENCODER_COUNTS_PER_REV * 4) * PIN_TO_SPUR_RATIO * 2 * np.pi * WHEEL_RADIUS)

    def execute(self, action: np.ndarray) -> np.ndarray:
        start_time: float = time.time()
        # TODO: counter for the number of steps we skip
        image: np.ndarray = self.front_csi.await_image()  # update image till image is not none
        # execute the control
        throttle, steering = super().execute(action)
        # print(f'steering: {steering}')
        self.update_state(image)
        # calculate the estimated speed
        linear_speed: float = self.estimate_speed()
        # calculate the sleep time
        execute_time: float = elapsed_time(start_time)
        sleep_time: float = self.monitor.dt - execute_time
        # realtime_message_output(f"Estimated speed: {linear_speed:1.2f}m/s, Real speed: {self.ego_state[3]:1.2f}m/s")
        time.sleep(sleep_time) if sleep_time > 0 else None
        # return the car's action to the environment
        return np.array([throttle, steering])


class ReinformerCar(WaypointCar):
    def setup(
        self, 
        waypoints: np.ndarray, 
        init_waypoint_index: int, 
        policy: ReinformerPolicy
    ) -> None:
        if policy is None:
            raise ValueError("Policy cannot be None")
        super().setup(waypoints, init_waypoint_index)
        self.policy: ReinformerPolicy = policy

    def handle_observation(self, orig: np.ndarray, rot: np.ndarray) -> None:
        super().handle_observation(orig, rot)
        far_index: int = (self.current_waypoint_index + 49) % self.waypoints.shape[0]
        self.observation['state'] = np.concatenate((
            self.ego_state, # ego state
            self.waypoints[self.current_waypoint_index], # closest waypoint
            self.waypoints[far_index] # far waypoint
        )) # state info

    def execute(self) -> None:
        # get the start time of the loop
        start_time: float = time.time()
        # get the action from the policy
        action, _ = self.policy.execute(self.observation)
        # apply the action to the vehicle
        self.running_gear.read_write_std(action[0] * 0.08, action[1])
        self.update_state()
        # sleep for the remaining time
        execute_time: float = elapsed_time(start_time)
        sleep_time: float = self.monitor.dt - execute_time
        time.sleep(max(0, sleep_time))