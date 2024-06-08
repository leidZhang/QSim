from typing import Tuple

import numpy as np

from core.control.pid_control.pid_controller import PIDController


class SteeringPIDController(PIDController):
    """
    The SteeringPIDController class is a class that generates the steering angle for the car
    based on the slope and intercept on the frame
    """

    def __init__(
            self,
            upper_bound: float,
            lower_bound: float,
            # slope_offset: float = DEFAULT_SLOPE_OFFSET,
            # intercept_offset: float = DEFAULT_INTERCEPT_OFFSET
        ) -> None:
        """
        Initializes the SteeringPIDController object

        Parameters:
        - upper_bound: float: The upper bound of the PID controller
        - lower_bound: float: The lower bound of the PID controller
        """
        super().__init__(upper_bound, lower_bound)
        # self.slope_offset: float = slope_offset
        # self.intercept_offset: float = intercept_offset

    def setup(self, k_p: float, k_i: float, k_d: float, offsets: Tuple[float, float]) -> None:
        """
        The setup method to set the PID gains

        Parameters:
        - k_p: float: The proportional gain of the PID controller
        - k_i: float: The integral gain of the PID controller
        - k_d: float: The derivative gain of the PID controller

        Returns:
        - None
        """
        self.slope_offset = offsets[0]
        self.intercept_offset = offsets[1]
        super().setup(k_p, k_i, k_d)

    def handle_control_error(self, slope: float, intercept: float, image_width: float) -> bool:
        """
        Calculate the cross error based on the slope and intercept

        Parameters:
        - slope: float: the slope of the lane
        - intercept: float: the intercept of the lane
        - image_width: the frame width

        Returns:
        - bool: whether we can handle the cross error
        """
        # fault tolerance
        if slope == 0.3419:
            return False
        if abs(slope) < 0.2 and abs(intercept) < 100:
            slope = self.slope_offset
            intercept = self.intercept_offset
        # calculate the cross error
        self.control_error: float = (intercept/-slope) - (self.intercept_offset / -self.slope_offset)
        self.control_error = self.control_error / image_width
        return True

    def execute(self, input: tuple, image_width: float) -> float:
        """
        The execute method to generate the steering angle based on the slope and intercept

        Parameters:
        - input: tuple: The input of the steering PID controller
        - image_width: float: The width of the image

        Returns:
        - float: The steering angle of the car
        """
        # decode the input
        slope: float = input[0]
        intercept: float = input[1]
        if self.handle_control_error(slope, intercept, image_width):
            return super().execute() # calculate the steering angle
        return 0.0