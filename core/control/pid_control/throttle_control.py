from typing import Any
from .pid_controller import PIDController


class ThrottlePIDController(PIDController):
    def __init__(self, upper_bound: float, lower_bound: float) -> None:
        super().__init__(upper_bound, lower_bound)
        self.throttle_offset: float = 0.0 # temp value

    def handle_cross_error(self, expected_speed: float, measeured_speed: float) -> Any:
        self.cross_error: float = expected_speed - measeured_speed

    def execute(self, expected_speed: float, measeured_speed: float) -> float:
        # calculate the cross error
        self.handle_cross_error(expected_speed, measeured_speed)
        return super().execute()