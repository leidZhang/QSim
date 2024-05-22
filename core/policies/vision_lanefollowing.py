from typing import Tuple

import numpy as np

from core.base_policy import BasePolicy
from core.control.vision_steering_control import VisionSteeringController
from core.control.close_loop_control import ThrottlePIDController


class VisionLaneFollowing(BasePolicy): 
    """
    The VisionLaneFollowing class is a class that generates the steering and throttle values for the car

    Attributes:
    - steering_controller: VisionSteeringController: The steering controller of the car
    - throttle_controller: ThrottlePIDController: The throttle controller of the car

    Methods:
    - setup_steering: Sets up the steering controller
    - setup_throttle: Sets up the throttle controller
    - execute: Executes the lane following
    """

    def __init__(self, image_width: float, image_height: float, ) -> None:
        """
        Initializes the VisionLaneFollowing object

        Parameters:
        - image_width: float: The width of the image
        - image_height: float: The height of the image

        Returns:
        - None
        """
        self.steering_controller = VisionSteeringController(image_width=image_width, image_height=image_height)
        self.throttle_controller = ThrottlePIDController(upper_bound=0.3, lower_bound=-0.3)
        
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

    def execute(self, observation: dict) -> Tuple[np.ndarray, dict]:
        """
        The execute method to generate the steering and throttle values based on the image and velocity

        Parameters:
        - observation: dict: The observation dictionary containing the image and velocity

        Returns:
        - Tuple[np.ndarray, dict]: The action and an empty info dictionary (as required by the 
        BasePolicy interface)
        """
        # get the image and velocity from the observation dict
        image: np.ndarray = observation['image']
        velocity: float = observation['state'][3] # state[3] is the velocity
        # get the steering and throttle values
        steering: float = self.steering_controller.execute(image)
        throttle: float = self.throttle_controller.execute(velocity) # pwm
        return np.array([throttle, steering]), {}
