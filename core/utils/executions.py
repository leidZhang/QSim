import os
import time
import logging
import warnings
from typing import Any, Dict, List, Union
from threading import Thread
from multiprocessing import Process
from threading import Event as ThEvent
from multiprocessing import Event as MpEvent
from abc import ABC, abstractmethod

from .performance import skip

Event = Union[ThEvent, MpEvent]


class WatchDogException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class WatchDogTimer:
    def __init__(self, event: Event, timeout: float = 0.1) -> None:
        self.event: Event = event
        self.timer: float = None
        self.timeout: float = timeout

    def start_watch_dog(self) -> None:
        # Wait for the event to be set for the first time
        self.event.wait()
        # Clear the watchdog event
        self.event.clear()
        # Initialize the timer
        self.timer = time.time()

    def check_timer(self) -> bool:
        # Ensure the watchdog has been started
        if self.timer is None:
            raise WatchDogException("WatchDogTimer not started. Please call start_watch_dog first.")

        # Check if the event is set
        if self.event.is_set():
            self.event.clear()
            self.timer = time.time()
            return True

        # Check if the timer has exceeded the timeout
        if time.time() - self.timer > self.timeout:
            return False

        # No time out and event set
        return True

    def set_timeout(self, timeout: float) -> None:
        if timeout <= 0:
            raise ValueError("Timeout must be a positive number.")
        self.timeout = timeout


class BaseCoordinator(ABC):
    def __init__(self, watchdogs: Dict[str, WatchDogTimer]) -> None:
        self.watchdogs: Dict[str, WatchDogTimer] = watchdogs
        self.pools: Dict[str, List[Union[Process, Thread]]] = {'thread': [], 'process': []}
        if self.watchdogs is not None:
            self.prepare_processes = self._setup_process_watchdogs
            self.prepare_threads = self._setup_thread_watchdogs
            self.execute_main_thread = self._monitor_system
        else:
            warnings.warn("The coordinator will not monitor the system!")
            self.prepare_processes = self._skip_monitoring
            self.prepare_threads = skip
            self.execute_main_thread = self._skip_monitoring

    def _skip_monitoring(self) -> None:
        time.sleep(8)

    def _setup_process_watchdogs(self) -> None:
        for _, watchdog in self.watchdogs['process'].items():
            watchdog.start_watch_dog()

    def _setup_thread_watchdogs(self) -> None:
        for _, watchdog in self.watchdogs['thread'].items():
            watchdog.start_watch_dog()

    def _monitor_system(self) -> None:
        for type in self.watchdogs.keys():
            for key, watchdog in self.watchdogs[type].items():
                if watchdog.check_timer():
                    continue
                raise WatchDogException(message=f"{key} time out")

    @abstractmethod
    def terminate(self) -> None:
        ...

    @abstractmethod
    def start_main_process(self) -> None:
        ...

    def run_monitor_thread(self) -> None:
        try:
            self.prepare_threads()
            while True:
                self.execute_main_thread()
                time.sleep(0.01)
        except KeyboardInterrupt:
            logging.info("Stopping System...")
        except WatchDogException as e:
            logging.error(e)
        except Exception as e:
            logging.critical(f"System stopped unexpectedly due to {e}")
        finally:
            self.terminate()
            logging.info("System stopped")
            os._exit(0)


class BaseProcessExec(ABC):
    def __init__(self, watchdog_event) -> None:
        # process event to control loop
        self.done = MpEvent()
        # watchdog event for monitoring
        if watchdog_event is not None:
            self.watchdog_event = watchdog_event # multi-processing Event
            self.reach_new_stage = self._set_watchdog
        else:
            warnings.warn("The process watchdog is unconfigured")
            self.reach_new_stage = skip

    def _set_watchdog(self) -> None:
        self.watchdog_event.set()

    def terminate(self) -> None:
        self.done.set()

    # TODO: May need to delete this and create a template
    def final(self) -> None:
        pass

    @abstractmethod
    def create_instance(self) -> Any:
        ...

    # TODO: May need to replace self.final() with instance.final()
    def run_process(self, *args) -> None:
        # create the instance of the process
        instance: Any = self.create_instance()
        # setup the watchdog or skip
        self.reach_new_stage()
        # main loop of the process
        while not self.done.is_set():
            # module executions
            instance.execute(*args)
            # set the watchdog or skip
            self.reach_new_stage()
        # final execution to close resources
        self.final()


class BaseThreadExec(ABC):
    def __init__(self, watchdog_event: ThEvent) -> None:
        # thread event for thread loop
        self.done: ThEvent = ThEvent()
        # watchdog event for monitoring
        if watchdog_event is not None:
            self.watchdog_event: ThEvent = watchdog_event
            self.reach_new_stage = self._set_watchdog
        else:
            warnings.warn("The thread watchdog is unconfigured")
            self.reach_new_stage = skip

    def _set_watchdog(self) -> None:
        self.watchdog_event.set()

    def terminate(self) -> None:
        self.done.set()

    def final(self) -> None:
        pass

    @abstractmethod
    def setup_thread(self) -> None:
        ...

    @abstractmethod
    def execute(self, *args: Any) -> None:
        ...

    def run_thread(self, *args: Any) -> None:
        # prepare the thread instances
        self.setup_thread()
        # setup the watchdog
        self.watchdog_event.set()
        # main loop of the thread
        while not self.done.is_set():
            self.execute(*args)
            # set the watchdog
            self.watchdog_event.set()
        # final execution to close resources
        self.final()
