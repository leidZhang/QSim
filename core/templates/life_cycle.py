from typing import Any
from abc import ABC, abstractmethod


class LifeCycleWrapper(ABC):
    """
    LifeCycleWrapper is an abstract class that defines the interface for 
    the life cycle of a thread or a process. 
    """
    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> None:
        """
        This method is responsible for executing the main logic of the 
        thread or process.
        """
        ...

    def terminate(self) -> None:
        """
        This method is responsible for terminating the modules in the thread 
        or process.
        """
        ...
        