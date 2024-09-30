from queue import Queue
from typing import Union


class RollingAverageTrigger:
    """
    RollingAverageTrigger is a class that implements a rolling average trigger.
    When the average of the data in the queue is greater than the threshold, 
    the event is set and the observer will handle the event.

    Attributes:
    - max_size: int: the maximum size of the queue
    - event: Event: the event to be triggered
    - threshold: float: the threshold for the trigger
    - accumulator: float: the accumulator for the rolling average
    - queue: Queue: the queue to store the history of the data
    """
    
    def __init__(self, event, threshold: float, size: int = 5) -> None:
        """
        Initializes the RollingAverageTrigger object.

        Parameters:
        - event (Event): The event to be triggered
        - threshold (float): The threshold for the trigger
        - size (int): The maximum size of the queue
        """
        self.max_size: int = size
        self.event = event
        self.threshold: float = threshold
        self.accumulator: float = 0.0
        self.queue: Queue = Queue(size)

    def clear(self) -> None:
        """
        Clears the queue and the accumulator.

        Returns:
        - None
        """
        self.queue = Queue(self.max_size)
        self.accumulator = 0.0

    def handle_moving_average(self, data: Union[float, int]) -> float:
        """
        Handles the moving average of the data.

        Parameters:
        - data (Union[float, int]): The data to be handled

        Returns:
        - float: The moving average of the data
        """
        if self.queue.full():
            self.accumulator -= self.queue.get()
        self.queue.put(data)
        self.accumulator += data
        return self.accumulator / self.max_size
    