from typing import Any
from threading import Event
from abc import abstractmethod


class BaseComm:
    def __init__(self, event) -> None:
        self.event: Event = event

    def terminate(self) -> None:
        self.event.set()

    @abstractmethod
    def execute(self, *args: Any) -> None:
        ...

    def run_comm(self, *args: Any) -> None:
        while not self.event.is_set():
            self.execute(*args)