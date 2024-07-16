import time
from queue import Queue
from typing import Any, Union
from threading import Event as MtEvent
from multiprocessing import Event as MpEvent

Event = Union[MtEvent, MpEvent]


class RollingAverageTrigger:
    def __init__(self, event: Event, threshold: float, size: int = 5) -> None:
        self.max_size: int = size
        self.event: Event = event
        self.threshold: float = threshold
        self.accumulator: float = 0.0
        self.queue: Queue = Queue(size)

    def clear(self) -> None:
        self.queue = Queue(self.max_size)
        self.accumulator = 0.0

    def handle_moving_average(self, data: Union[float, int]) -> float:
        if self.queue.full():
            self.accumulator -= self.queue.get()
        self.queue.put(data)
        self.accumulator += data
        return self.accumulator / self.max_size


class StopSignTrigger(RollingAverageTrigger):
    def __init__(self, event: Event, threshold: float, size: int = 5) -> None:
        super().__init__(event, threshold, size)
        self.last_stop_time: float = 0.0

    def __call__(self, data: Union[float, int]) -> None:
        res: float = self.handle_moving_average(data)
        if res > self.threshold and time.time() - self.last_stop_time >= 5:
            self.event.set()


class TrafficLightTrigger(RollingAverageTrigger):
    def __call__(self, data: Union[float, int]) -> None:
        res: float = self.handle_moving_average(data) / 2
        if not self.event.is_set() and res >= self.threshold:
            self.event.set()
        elif self.event.is_set() and res < self.threshold:
            self.event.clear()
