from typing import Any
from .pid_controller import PIDController


class ThrottlePIDController(PIDController):
    def __init__(self, upper_bound: float, lower_bound: float) -> None:
        super().__init__(upper_bound, lower_bound)
        self.throttle_offset: float = 0.0 # temp value

    def handle_control_error(self, expected_speed: float, measured_speed: float) -> Any:
        self.control_error: float = expected_speed - measured_speed

    def execute(self, expected_speed: float, measured_speed: float) -> float:
        # calculate the cross error
        self.handle_control_error(expected_speed, measured_speed)
        return super().execute()