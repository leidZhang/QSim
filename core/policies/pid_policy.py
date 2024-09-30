import time
import math
from typing import Tuple

import numpy as np

from core.control.pid_control import PIDController
from core.control.pid_control import SteeringPIDController, ThrottlePIDController
from core.templates.base_policy import BasePolicy


class CompositePIDPolicy(BasePolicy):
    """
    The CompositePIDPolicy class is a class that generates the steering and throttle values for the car
    based on the input data from the edgefinder
    """

    def __init__(self, expected_velocity: float) -> None:
        """
        Initializes the CompositePIDPolicy object

        Parameters:
        - expected_velocity: float: The expected velocity for the QCar
        - edge_finder: TraditionalEdgeFinder: The edge finder class for edge detection
        """
        self.reference_velocity: float = 0.0
        self.expected_velocity: float = expected_velocity
        self.steering_controller: PIDController = SteeringPIDController(upper_bound=0.5, lower_bound=-0.5)
        self.throttle_controller: PIDController = ThrottlePIDController(upper_bound=0.3, lower_bound=0.0)

    def setup_steering(self, k_p: float, k_i: float, k_d: float, offsets: Tuple[float, float]) -> None:
        """
        Sets up the steering controller

        Parameters:
        - k_p: float: The proportional gain of the PID controller
        - k_i: float: The integral gain of the PID controller
        - k_d: float: The derivative gain of the PID controller

        Returns:
        - None
        """
        self.steering_controller.setup(k_p, k_i, k_d, offsets)

    def setup_throttle(self, k_p: float, k_i: float, k_d: float) -> None:
        """
        Sets up the throttle controller

        Parameters:
        - k_p: float: The proportional gain of the PID controller
        - k_i: float: The integral gain of the PID controller
        - k_d: float: The derivative gain of the PID controller

        Returns:
        - None
        """
        self.throttle_controller.setup(k_p, k_i, k_d)

    def reset_start_time(self) -> None:
        """
        Resets the start time of the controllers
        """
        self.steering_controller.start = time.time()
        self.throttle_controller.start = time.time()

    def reset_delta_t(self) -> None:
        """
        Resets the delta time of the controllers
        """
        self.steering_controller.dt = 0.0
        self.throttle_controller.dt = 0.0

    def execute(self, steering_input: tuple, linear_speed: float, reduce_factor: float = 1.0) -> Tuple[np.ndarray, dict]:
        """
        The execute method to generate the steering and throttle values based on the image and velocity

        Parameters:
        - steering_input: np.ndarray: The image from the front csi camera
        - linear_speed: float: The velocity of the car, calculated from the encoder

        Returns:
        - Tuple[np.ndarray, dict]: The action and an empty info dictionary (as required by the
        BasePolicy interface)
        """
        # get the steering and throttle values
        self.steering: float = self.steering_controller.execute(steering_input[:2], steering_input[2])
        self.reference_velocity = self.expected_velocity * abs(math.cos(2.7 * self.steering)) * reduce_factor
        self.throttle: float = self.throttle_controller.execute(self.reference_velocity, linear_speed)
        # print(f'Expected: {self.reference_velocity:1.4f}, Measured:{linear_speed:1.4f}, dt: {self.throttle_controller.dt:1.4f}')

        return np.array([abs(self.throttle), self.steering]), {}