import numpy as np


def get_yaw_noise(action: np.ndarray, epsilon: float = 0.31) -> float:
    random: int = np.random.randint(0, 3)
    if random != 1:
         return 0.0
    
    rand_action_yaw = np.random.rand(1) * 2 - 1  # Generate random actions between -1 and 1
    if np.random.uniform(0, 1) < epsilon:
        return (rand_action_yaw - action[1])[0]
    return 0.0