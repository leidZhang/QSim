import time
from typing import Tuple

import numpy as np

from pal.utilities.math import Calculus
from qvl.qlabs import QuanserInteractiveLabs

from core.qcar import VirtualCar
from core.utils.tools import elapsed_time
from constants import MAX_LOOKAHEAD_INDICES


class WaypointCar(VirtualCar):
    def __init__(
            self,
            actor_id: int,
            dt: float,
            qlabs: QuanserInteractiveLabs,
            throttle_coeff: float = 0.3,
            steering_coeff: float = 0.5
        ) -> None:
        super().__init__(actor_id, dt, qlabs, throttle_coeff, steering_coeff)
        self.observation: dict = {} # observation dict
        self.diff = Calculus().differentiator_variable(dt)
        _ = next(self.diff)

    def setup(self, waypoints: np.ndarray) -> None:
        self.waypoints: np.ndarray = waypoints
        self.ego_state: np.ndarray = self.get_ego_state()
        orig, yaw, rot = self.cal_vehicle_state(self.ego_state)
        self.current_waypoint_index: int = 0
        self.next_waypoints: np.ndarray = self.waypoints
        self.observation["waypoints"] = np.matmul(self.next_waypoints[:MAX_LOOKAHEAD_INDICES] - orig, rot)

    def update_state(self) -> None:
        self.ego_state = self.get_ego_state()
        orig, yaw, rot = self.cal_vehicle_state(self.ego_state)
        local_waypoints: np.ndarray = np.roll(self.waypoints, -self.current_waypoint_index, axis=0)[:200]
        self.norm_dist: np.ndarray = np.linalg.norm(local_waypoints - orig, axis=1)
        self.dist_ix: int = np.argmin(self.norm_dist)
        self.current_waypoint_index = (self.current_waypoint_index + self.dist_ix) % self.waypoints.shape[0]
        self.next_waypoints = self.next_waypoints[self.dist_ix:] # clear pasted waypoints
        # in case of less than 200 waypoints
        if self.next_waypoints.shape[0] < MAX_LOOKAHEAD_INDICES:
            slop = MAX_LOOKAHEAD_INDICES - self.next_waypoints.shape[0]
            self.next_waypoints = np.concatenate([self.next_waypoints, self.waypoints[:slop]])
        # add waypoints info to observation
        self.observation['waypoints'] = np.matmul(self.next_waypoints[:MAX_LOOKAHEAD_INDICES] - orig, rot)

    def execute(self, action: np.ndarray) -> np.ndarray:
        start_time: float = time.time()
        # execute the control
        throttle, steering = super().execute(action)
        self.update_state()
        # plot estimated speed
        motor_tach: float = self.running_gear.motorTach
        self.estimate_speed(motor_tach)
        # calculate the sleep time
        end: float = elapsed_time(start_time)
        sleep_time: float = self.monitor.dt - end
        time.sleep(sleep_time if sleep_time > 0 else 0) # sleep dt every iter
        return np.array([throttle, steering])
