from typing import Any
from .pid_controller import PIDController


class ThrottlePIDController(PIDController):
    """
    The ThrottlePIDController class is a class that implements a PID controller for the throttle.

    Attributes:
    - throttle_offset: float, default=0.0
    """
    
    def __init__(self, upper_bound: float, lower_bound: float) -> None:
        """
        ThrottlePIDController class constructor, initialize the controller with the given parameters.

        Parameters:
        - upper_bound: float: The upper bound of the controller
        - lower_bound: float: The lower bound of the controller
        """
        super().__init__(upper_bound, lower_bound)
        self.throttle_offset: float = 0.0 # temp value

    def handle_control_error(self, expected_speed: float, measured_speed: float) -> Any:
        """
        Handle the control error by calculating the difference between the expected speed and the measured speed.

        Parameters:
        - expected_speed: float: The expected speed
        - measured_speed: float: The measured speed
        """
        self.control_error: float = expected_speed - measured_speed

    def execute(self, expected_speed: float, measured_speed: float) -> float:
        """
        Calculate the cross error and execute the controller.

        Parameters:
        - expected_speed: float: The expected speed
        - measured_speed: float: The measured speed

        Returns:
        - float: The throttle value
        """
        # calculate the cross error
        self.handle_control_error(expected_speed, measured_speed)
        return super().execute()