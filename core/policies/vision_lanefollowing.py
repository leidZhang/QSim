import math
from typing import Tuple

import numpy as np

from core.control.edge_finder import EdgeFinder
from core.utils.performance import realtime_message_output
from .pid_policy import CompositePIDPolicy


class VisionLaneFollowing(CompositePIDPolicy):
    """
    The VisionLaneFollowing class is a class that generates the steering and throttle values for the car
    based on the input image
    """

    def __init__(self, expected_velocity: float, edge_finder: EdgeFinder, image_width: int = 820) -> None:
        """
        Initializes the VisionLaneFollowing object

        Parameters:
        - expected_velocity: float: The expected velocity for the QCar
        - edge_finder: TraditionalEdgeFinder: The edge finder class for edge detection
        """
        self.edge_finder: EdgeFinder = edge_finder # TraditionalEdgeFinder(image_width, image_height)
        super().__init__(expected_velocity=expected_velocity, image_width=image_width)

    def execute(self, image: np.ndarray, linear_speed: float, reduce_factor: float = 1.0) -> Tuple[np.ndarray, dict]:
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
        # print(f'Expected: {self.reference_velocity:1.4f}, Measured:{linear_speed:1.4f}, dt: {self.throttle_controller.dt:1.4f}')

        return np.array([abs(self.throttle), self.steering]), {}
