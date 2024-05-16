from typing import Union, Tuple

import numpy as np

from pal.products.qcar import QCar

# from core.sensor.sensor import VirtualCSICamera, VirtualRGBDCamera
from core.base_policy import PolicyAdapter, BasePolicy
from .monitor import Monitor


class BaseCar: 
    def __init__(self, actor_id, dt, throttle_coeff: float = 0.3, steering_coeff: float = 0.5) -> None:
        # basic attributes
        self.actor_id: int = actor_id
        self.running_gear: QCar = QCar(id=actor_id)
        self.monitor: Monitor = Monitor(160, actor_id, dt=dt)
        self.throttle_coeff: float = throttle_coeff
        self.steering_coeff: float = steering_coeff
        # custom attributes
        self.policy = None

    def set_policy(self, policy: Union[PolicyAdapter, BasePolicy]) -> None: 
        self.policy = policy

    def get_ego_state(self) -> np.ndarray:
        return self.monitor.get_state()
    
    def get_vehicle_state(self, ego_state: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        orig: np.ndarray = ego_state[:2]
        yaw: float = -ego_state[2]
        rot: np.ndarray = np.array([
            [np.cos(yaw), np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)]
        ])
        return orig, yaw, rot

    def execute(self, observation: dict) -> None: 
        action, metrics = self.policy.execute(observation)
        throttle: float = self.throttle_coeff * action[0]
        steering: float = self.steering_coeff * action[1]
        self.running_gear.read_write_std(throttle, steering)
        return action, metrics