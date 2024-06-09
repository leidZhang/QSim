from typing import Any
from threading import Event as ThEvent
from multiprocessing import Event as MpEvent
from abc import ABC, abstractmethod


class BaseProcessExec(ABC):
    def __init__(self) -> None:
        self.event = MpEvent() # process event

    def terminate(self) -> None:
        self.event.set()

    @abstractmethod
    def create_instance(self) -> Any:
        ...

    def run_process(self, *args) -> None:
        instance: Any = self.create_instance()
        while not self.event.is_set():
            instance.execute(*args)


class BaseThreadExec(ABC):
    def __init__(self) -> None:
        self.event: ThEvent = ThEvent()

    def terminate(self) -> None:
        self.event.set()

    @abstractmethod
    def setup_thread(self) -> None:
        ...

    @abstractmethod
    def execute(self, *args: Any) -> None:
        ...

    def run_thread(self, *args: Any) -> None:
        self.setup_thread()
        while not self.event.is_set():
            self.execute(*args)
