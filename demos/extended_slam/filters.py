from queue import Queue
from typing import Any, Union
from threading import Event


class RollingAverageFilter:
    def __init__(self, event: Event, threshold: float, size: int = 5) -> None:
        self.max_size: int = size
        self.event: Event = event
        self.threshold: float = threshold
        self.accumulator: float = 0.0
        self.queue: Queue = Queue(size)

    def clear(self) -> None:
        self.queue = Queue(self.max_size)
        self.accumulator = 0.0

    def handle_moving_average(self, data: Union[float, int]) -> None:
        if self.queue.full():
            self.accumulator -= self.queue.get()
        self.queue.put(data)
        self.accumulator += data


class StopSignTrigger(RollingAverageFilter):
    def __call__(self, data: Union[float, int], additional_cond: bool) -> None:
        super().handle_moving_average(data)
        if self.accumulator / self.max_size > self.threshold and additional_cond:
            self.event.set()


class TrafficLightTrigger(RollingAverageFilter):
    def __call__(self, data: Union[float, int]) -> Any:
        super().handle_moving_average(data)
        if self.accumulator / self.max_size > self.threshold:
            return True
