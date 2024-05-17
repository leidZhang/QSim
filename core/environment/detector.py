from queue import Queue

import numpy as np

from .exception import AnomalousEpisodeException


class EpisodeMonitor:
    def __init__(self, start_orig: np.ndarray) -> None:
        self.queue: Queue = Queue(10)
        self.start_orig: np.ndarray = start_orig
        self.accumulator: int = 0
        print("aaa")

    def __call__(self, action: np.ndarray, orig: np.ndarray) -> None:
        result: int = 1
        if action[0] >= 0.045 and np.array_equal(orig, self.start_orig):
            result: int = 0

        if self.queue.full():
            self.accumulator -= self.queue.get()
        self.queue.put(result)
        self.accumulator += result

        if self.queue.full() and self.accumulator == 0:
            raise AnomalousEpisodeException("Error happened in the episode!")
