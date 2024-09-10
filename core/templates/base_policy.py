from typing import Any, Tuple
from abc import ABC, abstractmethod

import numpy as np


class BasePolicy(ABC):
    """
    The BasePolicy class is an abstract class that defines the interface for
    the control policies
    """

    def __init__(self, *args) -> None:
        """
        Initializes the BasePolicy object

        Parameters:
        - args: Any: The arguments of the policy

        Returns:
        - None
        """
        ...

    @abstractmethod
    def execute(self, *args) -> Tuple[np.ndarray, dict]:
        """
        The execute method is an abstract method that executes the policy

        Parameters:
        - observation: dict: The observation of the environment

        Returns:
        - Tuple[dict, dict]: The action and metrics to be taken by the agent
        """
        ...


class PolicyAdapter(ABC):
    """
    The PolicyAdapter class is an abstract class that defines the interface for
    those policies that does not use the standard interface of the BasePolicy

    Attributes:
    - policy: Any: The policy to be adapted
    """

    def __init__(self, policy: Any) -> None:
        """
        Initializes the PolicyAdapter object

        Parameters:
        - policy: Any: The policy to be adapted

        Returns:
        - None
        """
        self.policy = policy

    @abstractmethod
    def execute(self, *args) -> Tuple[np.ndarray, dict]:
        """
        The execute method is an abstract method that executes the policy

        Parameters:
        - observation: dict: The observation of the environment

        Returns:
        - Tuple[dict, dict]: The action and metrics to be taken by the agent
        """
        ...
