import time
import math
from typing import Dict, Tuple

from core.qcar import PhysicalCar
from core.control.pid_control import PIDController
from core.control.pid_control import SteeringPIDController, ThrottlePIDController


class PIDControlCar(PhysicalCar):
    def __init__(self, throttle_coeff: float, steering_coeff: float, desired_speed: float = 0.70) -> None:
        super().__init__(throttle_coeff, steering_coeff)
        self.desired_speed: float = desired_speed
        self.steering_controller: PIDController = SteeringPIDController(0.5, -0.5)
        self.throttle_controller: PIDController = ThrottlePIDController(0.3, 0)

    def setup(self, pid_gains: Dict[str, list], offsets: Tuple[float, float]) -> None:
        steering_gains: list = pid_gains['steering']
        throttle_gains: list = pid_gains['throttle']
        self.throttle_controller.setup(k_p=throttle_gains[0], k_i=throttle_gains[1], k_d=throttle_gains[2])
        self.steering_controller.setup(k_p=steering_gains[0], k_i=steering_gains[1], k_d=steering_gains[2], offsets=offsets)
        self.steering_controller.start = time.time()
        self.throttle_controller.start = time.time()

    def terminate(self) -> None:
        self.running_gear.terminate()

    # TODO: Implement this method
    def handle_events(self, stop_flags: list) -> bool:
        return False

    # TODO: Implement the event handle
    def execute(self, line_tuple: tuple, image_width: float, stop_flags: list, reverse_flag: bool) -> None:
        # temp if statement
        if not self.handle_events(stop_flags):
            estimated_speed: float = self.estimate_speed()
            steering: float = self.steering_controller.execute(line_tuple, image_width)
            reference_speed: float = self.desired_speed * abs(math.cos(2.7 * steering))
            throttle: float = self.throttle_controller.execute(
                expected_speed=reference_speed, 
                measured_speed=estimated_speed
            ) * (1 if reverse_flag else -1) # calculate the pwm value
            self.handle_leds(throttle=throttle, steering=steering)
            self.running_gear.read_write_std(throttle=throttle, steering=steering, LEDs=self.leds)
        else: 
            print("Stopping the car")
            self.halt_car(steering=steering, stop_time=3.0) # temp handle
