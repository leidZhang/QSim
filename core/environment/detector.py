from queue import Queue
from typing import List
from shapely import Polygon

import numpy as np

from .exception import AnomalousEpisodeException


class EpisodeMonitor:
    """
    EpisodeMonitor is a class that monitors the episode and raises an
    exception if the communication between the server and the client is
    anomalous.

    Attributes:
    - queue (Queue): The queue to store the results of the episode
    - start_orig (np.ndarray): The original start state of the episode
    - accumulator (int): The accumulator to store the results of the episode
    - threshold (float): The threshold to determine the anomaly
    """

    def __init__(self, start_orig: np.ndarray, threshold: float = 0.045) -> None:
        """
        Initializes the EpisodeMonitor object.

        Parameters:
        - start_orig (np.ndarray): The original start state of the episode
        - threshold (float): The threshold to determine the anomaly
        """
        self.queue: Queue = Queue(10)
        self.start_orig: np.ndarray = start_orig
        self.threshold: float = threshold
        self.accumulator: int = 0

    def __call__(self, action: np.ndarray, orig: np.ndarray) -> None:
        """
        Executes the EpisodeMonitor object.

        Parameters:
        - action (np.ndarray): The action taken in the episode
        - orig (np.ndarray): The original state of the episode

        Returns:
        - None

        Raises:
        - AnomalousEpisodeException: If the episode is anomalous
        """
        result: int = 1
        if action[0] >= self.threshold and np.array_equal(orig, self.start_orig):
            result: int = 0

        if self.queue.full():
            self.accumulator -= self.queue.get()
        self.queue.put(result)
        self.accumulator += result

        if self.queue.full() and self.accumulator == 0:
            raise AnomalousEpisodeException("Error happened in the episode!")

REAR_LENGTH: float = 0.06
FORWARD_LENGTH: float = 0.365
HALF_WIDTH: float = 0.095


class EnvQCarRef:
    """
    EnvQCarRef is a class that represents the reference bounding box of the QCar in the 
    environment.

    Attributes:
    - box (np.ndarray): The bounding box of the QCar
    """
    
    def __init__(self) -> None:
        """
        The constructor of the EnvQCarRef class. Initialize the bounding box of the QCar.
        """
        self.box: np.ndarray = np.array([
            [-HALF_WIDTH, -REAR_LENGTH],
            [HALF_WIDTH, -REAR_LENGTH],
            [HALF_WIDTH, FORWARD_LENGTH],
            [-HALF_WIDTH, FORWARD_LENGTH]
        ])

    def get_poly(self, state: np.ndarray) -> Polygon:
        """
        Get the polygon of the QCar based on the given state.

        Parameters:
        - state(np.ndarray): The state of the QCar

        Returns:
        - Polygon: The polygon of the QCar
        """
        x, y, yaw = state[:3]
        rot: np.ndarray = np.array([
            [-np.sin(yaw), np.cos(yaw)],
            [np.cos(yaw), np.sin(yaw)]
        ])
        corrected_box: np.ndarray = np.dot(self.box, rot) + np.array([x, y])
        return Polygon(corrected_box)


def is_collided(
    state_1: np.ndarray,
    state_2: np.ndarray,
    actor_type: EnvQCarRef,
    threshold: float = 0.0004,
) -> bool:
    """
    Check if two actors are collided based on the given states. If the distance between the two
    actors is greater than the forward length, then the two actors are not collided. Otherwise,
    use the polygon intersection to check if the two actors are collided.

    Parameters:
    - state_1(np.ndarray): The state of the first actor
    - state_2(np.ndarray): The state of the second actor
    - actor_type(EnvQCarRef): The type of the actor
    - threshold(float): The threshold to determine the collision

    Returns:
    - bool: True if the two actors are collided, False otherwise
    """
    if np.linalg.norm(state_1[:2] - state_2[:2]) > FORWARD_LENGTH * 2:
        return False

    poly_1: Polygon = actor_type.get_poly(state_1)
    poly_2: Polygon = actor_type.get_poly(state_2)
    if poly_1.intersects(poly_2):
        intersection_area: float = poly_1.intersection(poly_2).area
        return intersection_area > threshold
    return False

def is_point_in_box(orig: np.ndarray, box: np.ndarray) -> bool:
    """
    Check if a given point is inside the AABB of another box.

    Parameters:
    - orig(np.ndarray): The point to check.
    - box(np.ndarray): The AABB to check against.

    Returns:
    - bool: True if the point is inside the AABB, False otherwise.
    """
    return box["min_x"] < orig[0] <= box["max_x"] and box["min_y"] < orig[1] <= box["max_y"]
