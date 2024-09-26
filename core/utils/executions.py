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

from core.templates import LifeCycleWrapper
from .performance import skip

Event = Union[ThEvent, MpEvent]


class WatchDogException(Exception):
    """
    The exception raised when the watchdog timer times out or the watchdog
    is not started before checking the timer.
    """

    def __init__(self, message: str) -> None:
        """
        Initialize the WatchDogException with the message.

        Parameters
        - message: str: The message to be displayed when the exception is raised.
        """
        super().__init__(message)


class WatchDogTimer:
    """
    The WatchDogTimer class is used to monitor the execution of a process or thread.
    """

    def __init__(self, event, timeout: float = 0.1) -> None:
        """
        Initialize the WatchDogTimer with the event and timeout.

        Parameters:
        - event: Event: The thread event or process to monitor.
        - timeout: float: The timeout for the watchdog timer.
        """
        self.event = event
        self.timer: float = None
        self.timeout: float = timeout

    def start_watch_dog(self) -> None:
        """
        Start the watchdog timer by waiting for the event to be set for the first
        time.

        Returns:
        - None
        """
        # Wait for the event to be set for the first time
        self.event.wait()
        # Clear the watchdog event
        self.event.clear()
        # Initialize the timer
        self.timer = time.time()

    def check_timer(self) -> bool:
        """
        Check the watchdog timer to see if the event is set or the timer has exceeded
        the timeout.

        Returns:
        - bool: True if the event is set or the timer has not exceeded the timeout,
        False otherwise.

        Raises:
        - WatchDogException: If the watchdog timer is not started before checking the timer.
        """
        # Ensure the watchdog has been started
        if self.timer is None:
            raise WatchDogException(
                "WatchDogTimer not started. Please call start_watch_dog first."
            )

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
        """
        Set the timeout for the watchdog timer.

        Parameters:
        - timeout: float: The timeout for the watchdog timer.

        Returns:
        - None

        Raises:
        - ValueError: If the timeout is not a positive number.
        """
        if timeout <= 0:
            raise ValueError("Timeout must be a positive number.")
        self.timeout = timeout


class BaseCoordinator(ABC):
    """
    BaseCoordinator is an abstract class that defines the methods for the coordinator,
    which is used to monitor the system using watchdog timers in the main thread, and
    manage the processes and threads.
    """

    def __init__(self, watchdogs: Dict[str, WatchDogTimer] = None) -> None:
        """
        Initialize the BaseCoordinator with the watchdogs.

        Parameters:
        - watchdogs: Dict[str, WatchDogTimer]: The watchdog timers for monitoring the system.
        """
        self.sys_done = MpEvent()
        self.watchdogs: Dict[str, Dict[str, WatchDogTimer]] = watchdogs
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
        """
        The time sleep to replace the monitoring of the system when the watchdogs
        are not configured.

        Returns:
        - None
        """
        time.sleep(8)

    def _setup_process_watchdogs(self) -> None:
        """
        Start the watchdog timers for the processes.

        Returns:
        - None
        """
        for _, watchdog in self.watchdogs['process'].items():
            watchdog.start_watch_dog()

    def _setup_thread_watchdogs(self) -> None:
        """
        Start the watchdog timers for the threads.

        Returns:
        - None
        """
        for _, watchdog in self.watchdogs['thread'].items():
            watchdog.start_watch_dog()

    def _monitor_system(self) -> None:
        """
        Monitor the threads and processes by checking the watchdog timers.

        Returns:
        - None

        Raises:
        - WatchDogException: If one of the watchdog timers time out.
        """
        for type in self.watchdogs.keys():
            for key, watchdog in self.watchdogs[type].items():
                if watchdog.check_timer():
                    continue
                raise WatchDogException(message=f"{key} time out")

    @abstractmethod
    def terminate(self) -> None:
        """
        The method to terminate the system, the implementation is dependent on the
        system, and should be implemented in the subclass.

        Returns:
        - None
        """
        ...

    @abstractmethod
    def start_main_process(self) -> None:
        """
        The method to start the main process of the system, the implementation is
        dependent on the system, and should be implemented in the subclass.

        Returns:
        - None
        """
        ...

    def run_monitor_thread(self) -> None:
        """
        Execute the main thread to monitor the system.

        Returns:
        - None

        Raises:
        - KeyboardInterrupt: If the system is stopped by the user.
        - WatchDogException: If one of the watchdog timers time out.
        - Exception: If the system stops unexpectedly.
        """
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
            # os._exit(0)


class BaseProcessExec(ABC):
    """
    Abstract base class for process execution management.

    This class encapsulates the entire lifecycle of a process execution, providing
    a framework for initializing, executing, and finalizing a process. It is designed
    to be extended by concrete classes that implement specific process execution strategies.
    """

    def __init__(self, done, watchdog_event = None) -> None:
        """
        Initialize the BaseProcessExec with the watchdog event.

        Parameters:
        - done: Event: The done event for terminating the process.
        - watchdog_event: Event: The watchdog event for monitoring the process.
        """
        # process event to control loop
        self.done = done # multi-processing Event
        # watchdog event for monitoring
        if watchdog_event is not None:
            self.watchdog_event = watchdog_event # multi-processing Event
            self.reach_new_stage = self._set_watchdog
        else:
            warnings.warn("The process watchdog is unconfigured")
            self.reach_new_stage = skip

    def _set_watchdog(self) -> None:
        """
        This method is used to set the watchdog event before entering the main loop
        of the process or at the end of each iteration.

        Returns:
        - None
        """
        self.watchdog_event.set()

    def terminate(self) -> None:
        """
        This method is used to terminate the main loop process by setting the done event.

        Returns:
        - None
        """
        self.done.set()

    # TODO: May need to delete this and create a template
    def final(self, instance: LifeCycleWrapper) -> None:
        """
        This method is used to finalize the process execution by closing resources. Users can
        override this method to implement custom finalization logic.

        Returns:
        - None
        """
        instance.terminate()

    @abstractmethod
    def create_instance(self) -> Any:
        """
        This method is used to create the main instance of the process. Users must override this
        method to return the instance of the process.

        Returns:
        - Any: The main instance of the process.
        """
        # TODO: May need to replace Any with the actual type of the instance
        ...

    # TODO: May need to replace self.final() with instance.final()
    def run_process(self, *args) -> None:
        """
        This method is used to run the whole lifecyle of the process. It initializes the process
        instance, sets the watchdog event as the start flag (or simply skip if the user did not
        set a watchdog) of the main process loop, executes the main loop of the process, and
        finalizes the process.

        Parameters:
        - args: Any: The arguments to pass to the process instance.

        Returns:
        - None
        """
        try:
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
        except KeyboardInterrupt:
            print("KeyboardInterrupt detected!")
        except Exception as e:
            logging.error(f"Error: {e}")
        finally:
            # final execution to close resources
            self.final(instance)


class BaseThreadExec(ABC):
    """
    The abstract base class for thread execution management.

    This class encapsulates the entire lifecycle of a thread execution, providing a framework
    for initializing, executing, and finalizing a thread. It is designed to be extended by
    concrete classes that implement specific thread execution strategies.
    """

    def __init__(self, done, watchdog_event: ThEvent = None) -> None:
        """
        Initialize the BaseThreadExec with the watchdog event.

        Parameters:
        - done: Event: The done event for terminating the thread.
        - watchdog_event: ThEvent: The watchdog event for monitoring the thread.
        """
        # thread event for thread loop
        self.done = done
        # watchdog event for monitoring
        if watchdog_event is not None:
            self.watchdog_event: ThEvent = watchdog_event
            self.reach_new_stage = self._set_watchdog
        else:
            warnings.warn("The thread watchdog is unconfigured")
            self.reach_new_stage = skip

    def _set_watchdog(self) -> None:
        """
        This method is used to set the watchdog event before entering the main loop
        of the thread or at the end of each iteration.

        Returns:
        - None
        """
        self.watchdog_event.set()

    def terminate(self) -> None:
        """
        This method is used to terminate the main loop thread by setting the done event.
        """
        self.done.set()

    def final(self) -> None:
        """
        This method is used to finalize the thread execution by closing resources. Users can
        override this method to implement custom finalization logic.

        Returns:
        - None
        """
        ...

    @abstractmethod
    def setup_thread(self) -> None:
        """
        This method is used to setup the required instances for the thread execution before
        entering the main loop of the thread.

        Returns:
        - None
        """
        ...

    @abstractmethod
    def execute(self, *args: Any) -> None:
        """
        This method is used to execute the main loop of the thread. Users must override this
        method to implement the main logic of the thread.

        Parameters:
        - args: Any: The arguments to pass to the thread instance.

        Returns:
        - None
        """
        ...

    def run_thread(self, *args: Any) -> None:
        """
        This method is used to run the whole lifecyle of the thread. It initializes the thread
        instances, sets the watchdog event as the start flag (or simply skip if the user did not
        set a watchdog) of the main thread loop, executes the main loop of the thread, and finalizes
        the thread.

        Parameters:
        - args: Any: The arguments to pass to the thread instance.

        Returns:
        - None
        """
        try:
            # prepare the thread instances
            self.setup_thread()
            # setup the watchdog
            self.reach_new_stage()
            # main loop of the thread
            while not self.done.is_set():
                self.execute(*args)
                # set the watchdog
                self.reach_new_stage()
        except KeyboardInterrupt:
            print("KeyboardInterrupt detected!")
        except Exception as e:
            logging.error(f"Error: {e}")
        finally:
            # final execution to close resources
            self.final()
