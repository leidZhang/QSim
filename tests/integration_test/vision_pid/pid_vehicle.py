from typing import Dict

import numpy as np

from core.qcar.vehicle import PhysicalCar

from core.base_policy import BasePolicy
from core.qcar.sensor import VirtualCSICamera
from core.policies.vision_lanefollowing import VisionLaneFollowing
from core.control.edge_finder import EdgeFinder, TraditionalEdgeFinder
from core.control.edge_finder import NoImageException, NoContourException


class VisionPIDTestCar(PhysicalCar):
    def __init__(self, throttle_coeff: float, steering_coeff: float) -> None:
        super().__init__(throttle_coeff, steering_coeff)
        self.front_csi: VirtualCSICamera = VirtualCSICamera(id=3)
        self.reduce_coeff: float = 1.0

    def setup(self, expected_velocity: float, pid_gains: Dict[str, list]) -> None: 
        steering_gains: list = pid_gains['steering']
        throttle_gains: list = pid_gains['throttle']
        edge_finder: EdgeFinder = TraditionalEdgeFinder(image_width=820, image_height=410)
        self.policy: BasePolicy = VisionLaneFollowing(edge_finder=edge_finder, expected_velocity=expected_velocity)
        self.policy.setup_steering(k_p=steering_gains[0], k_i=steering_gains[1], k_d=steering_gains[2])
        self.policy.setup_throttle(k_p=throttle_gains[0], k_i=throttle_gains[1], k_d=throttle_gains[2])

    def estimate_speed(self) -> float:
        return float(self.running_gear.motorTach)
    
    def execute(self) -> None: 
        try: 
            image: np.ndarray = self.front_csi.read_image()
            if image is not None: 
                linear_speed: float = self.estimate_speed()
                action, _ = self.policy.execute(image, linear_speed, self.reduce_coeff)
                self.running_gear.read_write_std(throttle=action[0], steering=action[1], LEDs=self.leds)
        except NoContourException:
            print("No contour detected")