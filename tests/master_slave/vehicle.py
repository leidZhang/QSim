import time
import math
from typing import Dict, Tuple

from core.qcar import PhysicalCar
from core.policies.base_policy import BasePolicy
from core.policies.pid_policy import CompositePIDPolicy


class PIDControlCar(PhysicalCar):
    def __init__(self, throttle_coeff: float, steering_coeff: float, desired_speed: float = 0.70) -> None:
        super().__init__(throttle_coeff, steering_coeff)
        self.policy: CompositePIDPolicy = CompositePIDPolicy(expected_velocity=desired_speed)

    def setup(self, pid_gains: Dict[str, list], offsets: Tuple[float, float]) -> None:
        steering_gains: list = pid_gains['steering']
        throttle_gains: list = pid_gains['throttle']
        self.policy.setup_throttle(k_p=throttle_gains[0], k_i=throttle_gains[1], k_d=throttle_gains[2])
        self.policy.setup_steering(k_p=steering_gains[0], k_i=steering_gains[1], k_d=steering_gains[2], offsets=offsets)
        self.policy.reset_start_time()

    def terminate(self) -> None:
        self.running_gear.terminate()

    # TODO: Implement this method
    def handle_events(self, stop_flags: list) -> bool:
        return False

    # TODO: Implement the event handle
    def execute(self, line_tuple: tuple, stop_flags: list) -> None:
        # temp if statement
        if not self.handle_events(stop_flags):
            # TODO: use last state to control steering
            self.line_tuple: tuple = line_tuple
            estimated_speed: float = self.estimate_speed()
            self.action, _ = self.policy.execute(steering_input=self.line_tuple, linear_speed=estimated_speed)
            self.handle_leds(throttle=self.action[0], steering=self.action[1])
            self.running_gear.read_write_std(throttle=self.action[0], steering=self.action[1], LEDs=self.leds)
        else:
            print("Stopping the car")
            self.halt_car(steering=self.action[1], stop_time=3.0) # temp handle
