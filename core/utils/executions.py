import os
import time
from typing import Any, Dict, List
from threading import Thread
from multiprocessing import Process
from threading import Event as ThEvent
from multiprocessing import Event as MpEvent
from abc import ABC, abstractmethod


class BaseCoordinator(ABC):
    def __init__(self) -> None:
        self.pools: Dict[str, List[Process | Thread]] = {
            'thread': [], # store threads only
            'process': [] # store processes only
        }

    @abstractmethod
    def terminate(self) -> None:
        ...

    @abstractmethod
    def start_main_process(self) -> None:
        ...

    def observe_keyboard_interrupt(self) -> None:
        try:
            while True:
                time.sleep(100)
        except KeyboardInterrupt:
            print("Stopping System...")
        except Exception as e:
            print(f"System stopped unexpectedly due to {e}")
        finally:
            self.terminate()
            print("System stopped")
            os._exit(0)


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
