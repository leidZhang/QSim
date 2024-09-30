import random

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


def get_noise_by_action(action: np.ndarray, epsilon: float = 0.31) -> np.ndarray:
    """
    Get the noise based on the action and epsilon.

    Parameters:
    - action (np.ndarray): The action to get the noise
    - epsilon (float): The epsilon value

    Returns:
    - np.ndarray: The noise action
    """
    rand_action = np.random.rand(*action.shape) * 2 - 1
    if random.uniform(0, 1) < epsilon:
        return rand_action
    return action