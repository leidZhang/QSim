import numpy as np


class MockPose:
    """
    MockPose class to represent the pose of the car. It is 
    used to mimic the pose of the car in the simulator.
    
    Attributes:
    - position_x (float): The x position of the car
    - position_y (float): The y position of the car
    - orientation (float): The orientation of the car
    - velocity_x (float): The x velocity of the car
    - velocity_y (float): The y velocity of the car
    - angular_speed (float): The angular speed of the car
    """

    def __init__(self, state: np.ndarray) -> None:
        """
        Initializes the MockPose object.

        Parameters:
        - state (np.ndarray): The state of the car

        Returns:
        - None
        """
        self.position_x: float = state[0] # x
        self.position_y: float = state[1] # y
        self.orientation: float = state[2] # theta
        self.velocity_x: float = state[3] # v_x
        self.velocity_y: float = state[4] # v_y
        self.angular_speed: float = state[5] # omega

    def __str__(self) -> str:
        """
        String representation of the MockPose object.

        Returns:
        - str: The string representation of the MockPose object
        """
        return (
            f"Position: ({self.position_x}, {self.position_y}), "
            f"Orientation: {self.orientation}, "
            f"Velocity: ({self.velocity_x}, {self.velocity_y}), "
            f"Angular Speed: {self.angular_speed}"
        )