from typing import Tuple

import numpy as np

from .controller import PIDController, SteeringPIDController
from .edge_finder import TraditionalEdgeFinder
from .constants import DEFAULT_SLOPE_OFFSET, DEFAULT_INTERCEPT_OFFSET


class VisionSteeringController: 
    """
    VisionLaneFollowing class is a class that generates the steering angle for the car

    Attributes:
    - slope_offset: float: The slope offset of the lane
    - intercept_offset: float: The intercept offset of the lane
    - edge_finder: TraditionalEdgeFinder: The edge finder of the lane following
    - pid_controller: PIDController: The PID controller of the lane following

    Methods:
    - setup: Sets up the PID controller
    - execute: Executes the lane following
    """
    
    def __init__(self, image_width: int = 820, image_height: int = 410) -> None:
        """
        Initializes the VisionLaneFollowing object

        Parameters:
        - image_width: int: The width of the image
        - image_height: int: The height of the image
        """
        # default values when on the straight lane
        self.slope_offset: float = DEFAULT_SLOPE_OFFSET
        self.intercept_offset: float = DEFAULT_INTERCEPT_OFFSET
        # submodules for the vision lane following
        self.edge_finder: TraditionalEdgeFinder = TraditionalEdgeFinder(image_width, image_height)
        self.pid_controller: PIDController = SteeringPIDController(upper_bound=0.5, lower_bound=-0.5)

    def setup(self, k_p: float, k_i: float, k_d: float) -> None: 
        """
        Initializes the VisionLaneFollowing object

        Parameters:
        - k_p: float: The proportional gain of the PID controller
        - k_i: float: The integral gain of the PID controller
        - k_d: float: The derivative gain of the PID controller
        """
        self.pid_controller.setup(k_p, k_i, k_d)

    def execute(self, original_image: np.ndarray) -> float:
        """
        The execute method to generate the steering angle based on the slope and intercept

        Parameters:
        - original_image: np.ndarray: The original image from the csi camera

        Returns:
        - float: The steering angle of the car
        """
        result: tuple = self.edge_finder.execute(original_image)
        return self.pid_controller.execute(input=result, image_width=original_image.shape[0])
