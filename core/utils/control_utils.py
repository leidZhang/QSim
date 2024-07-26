import numpy as np


def get_yaw_noise(action: np.ndarray, epsilon: float = 0.31) -> float:
    """
    Get the yaw noise based on the action and epsilon.

    Parameters:
    - action (np.ndarray): The action to get the yaw noise
    - epsilon (float): The epsilon value

    Returns:
    - float: The yaw noise
    """
    rand_action_yaw = np.random.rand(1) * 2 - 1  # Generate random actions between -1 and 1
    if np.random.uniform(0, 1) < epsilon:
        return (rand_action_yaw - action[1])[0]
    return 0.0