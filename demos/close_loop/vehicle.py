import time
import math
from typing import Tuple

import numpy as np

from core.qcar.vehicle import PhysicalCar
from pal.utilities.math import Calculus

from core.qcar.sensor import VirtualCSICamera
from core.utils.tools import realtime_message_output, elapsed_time
from core.control.edge_finder import EdgeFinder, TraditionalEdgeFinder
from core.control.edge_finder import NoImageException, NoContourException
from core.control.pid_control import ThrottlePIDController, PIDController, SteeringPIDController
from core.qcar.constants import WHEEL_RADIUS, ENCODER_COUNTS_PER_REV, PIN_TO_SPUR_RATIO


class ThrottlePIDTestCar(PhysicalCar): # test frequency is 30Hz
    def __init__(self, throttle_coeff: float, steering_coeff: float) -> None:
        super().__init__(throttle_coeff, steering_coeff)
        self.diff = Calculus().differentiator_variable(0.03)
        _ = next(self.diff)
        self.throttle_control: PIDController = ThrottlePIDController(0.3, 0)
        self.expected_speed: float = 0.73
        self.counter: int = 0
    
    def setup(self, k_p: float, k_i: float, k_d: float):
        self.throttle_control.setup(k_p, k_i, k_d)

    def estimate_speed(self) -> float:
        encoder_counts: np.ndarray = self.running_gear.motorEncoder
        encoder_speed: float = self.diff.send((encoder_counts[0], 0.03))
        return encoder_speed * (1 / (ENCODER_COUNTS_PER_REV * 4) * PIN_TO_SPUR_RATIO * 2 * np.pi * WHEEL_RADIUS)

    def execute(self) -> None:
        start_time: float = time.time()
        estimated_speed: float = self.estimate_speed()
        self.throttle: float = self.throttle_control.execute(self.expected_speed, estimated_speed)
        self.running_gear.read_write_std(self.throttle, 0)
        end: float = elapsed_time(start_time)
        if self.counter % 2 == 0:
            output_message: str = f"Estimated speed: {estimated_speed:1.4f}, Expected speed: {self.expected_speed:1.4f}, Throttle: {self.throttle:1.4f} {' ' * 10}"
            realtime_message_output(output_message)
        sleep_time: float = (0.03 - end) if end < 0.03 else 0
        time.sleep(sleep_time)
        self.counter += 1


class VSPIDTestCar(PhysicalCar): 
    def __init__(self, throttle_coeff: float, steering_coeff: float) -> None:
        super().__init__(throttle_coeff, steering_coeff)
        self.front_csi: VirtualCSICamera = VirtualCSICamera(id=3)
        self.edge_finder: EdgeFinder = TraditionalEdgeFinder()
        self.steering_control: PIDController = SteeringPIDController(0.5, -0.5)

    def setup(self, k_p: float, k_i: float, k_d: float):
        self.steering_control.setup(k_p, k_i, k_d)
    
    def execute(self) -> None:
        try: 
            image: np.ndarray = self.front_csi.read_image()
            if image is not None: 
                result: Tuple[float, float] = self.edge_finder.execute(image)
                steering: float = self.steering_control.execute(input=result, image_width=image.shape[1])
                throttle: float = 0.18 * abs(math.cos(2.7 * steering))
                self.running_gear.read_write_std(throttle=throttle, steering=steering, LEDs=self.leds)
        except NoContourException:
            print("No contour detected")