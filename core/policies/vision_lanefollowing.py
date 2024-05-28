import math
import time
from typing import Tuple

import numpy as np

from core.policies.base_policy import BasePolicy
from core.control.edge_finder import EdgeFinder
from core.control.pid_control import ThrottlePIDController, SteeringPIDController, PIDController


class VisionLaneFollowing(BasePolicy):
    """
    The VisionLaneFollowing class is a class that generates the steering and throttle values for the car

    Attributes:
    - edge_finder: TraditionalEdgeFinder: The edge finder of the car
    - steering_controller: PIDController: The steering controller of the car
    - throttle_controller: PIDController: The throttle controller of the car

    Methods:
    - setup_steering: Sets up the steering controller
    - setup_throttle: Sets up the throttle controller
    - reset_start_time: Resets the start time of the controllers
    - reset_delta_t: Resets the delta time of the controllers
    - execute: Executes the lane following
    """

    def __init__(self, expected_velocity: float, edge_finder: EdgeFinder) -> None:
        """
        Initializes the VisionLaneFollowing object

        Parameters:
        - expected_velocity: float: The expected velocity for the QCar
        - edge_finder: TraditionalEdgeFinder: The edge finder class for edge detection

        Returns:
        - None
        """
        self.expected_velocity: float = expected_velocity # temporarly used the static speed
        self.reference_velocity: float = 0.0
        self.edge_finder: EdgeFinder = edge_finder # TraditionalEdgeFinder(image_width, image_height)
        self.steering_controller: PIDController = SteeringPIDController(upper_bound=0.5, lower_bound=-0.5)
        self.throttle_controller: PIDController = ThrottlePIDController(upper_bound=0.3, lower_bound=0)

    def setup_steering(self, k_p: float, k_i: float, k_d: float) -> None:
        """
        Sets up the steering controller

        Parameters:
        - k_p: float: The proportional gain of the PID controller
        - k_i: float: The integral gain of the PID controller
        - k_d: float: The derivative gain of the PID controller

        Returns:
        - None
        """
        self.steering_controller.setup(k_p, k_i, k_d)

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

    def execute(self, image: np.ndarray, linear_speed: float, reduce_factor: float) -> Tuple[np.ndarray, dict]:
        """
        The execute method to generate the steering and throttle values based on the image and velocity

        Parameters:
        - image: np.ndarray: The image from the front csi camera
        - linear_speed: float: The velocity of the car, calculated from the encoder

        Returns:
        - Tuple[np.ndarray, dict]: The action and an empty info dictionary (as required by the
        BasePolicy interface)
        """
        # get the steering and throttle values
        result: tuple = self.edge_finder.execute(image)
        self.steering: float = self.steering_controller.execute(result, image.shape[1])
        self.reference_velocity = self.expected_velocity * abs(math.cos(2.7 * self.steering)) * reduce_factor
        self.throttle: float = self.throttle_controller.execute(self.reference_velocity, linear_speed)
        # realtime_message_output(f'Expected: {self.reference_velocity:1.4f}, Measured:{linear_speed:1.4f}')

        return np.array([self.throttle, self.steering]), {}
