from queue import Queue

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
