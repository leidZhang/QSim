import numpy as np


class MockPose:
    def __init__(self, state: np.ndarray) -> None:
        self.position_x: float = state[0] # x
        self.position_y: float = state[1] # y
        self.orientation: float = state[2] # theta
        self.velocity_x: float = state[3] # v_x
        self.velocity_y: float = state[4] # v_y
        self.angular_speed: float = state[5] # omega

    def __str__(self) -> str:
        return (
            f"Position: ({self.position_x}, {self.position_y}), "
            f"Orientation: {self.orientation}, "
            f"Velocity: ({self.velocity_x}, {self.velocity_y}), "
            f"Angular Speed: {self.angular_speed}"
        )