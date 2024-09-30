import math
import warnings
from abc import abstractmethod
from typing import Tuple, List

from gym import Env
import numpy as np

from core.roadmap import ACCRoadMap
from .simulator import QLabSimulator

DEFAULT_MAX_STEPS: int = 1000
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseQLabEnv(Env):
    """
    Base class for QLab Environment, which defines the basic structure
    of the environment.
    """

    def _init_step_params(self) -> Tuple[dict, float, dict]:
        """
        Initialize the parameters for each step in the environment.

        Returns:
        - observation (dict): The observation of the environment
        - reward (float): The reward of the environment
        - info (dict): The information of the environment
        """
        observation: dict = {}
        reward: float = 0.0
        info: dict = {}
        return observation, reward, info

    def handle_reward(self, *args) -> Tuple[float, bool]:
        """
        This method handles the reward calculation for the environment.
        The function should return the reward and the done flag, and is
        expected to be implemented by the subclass.
        """
        return 0, False

    def step(self, *args) -> Tuple[dict, float, bool, dict]:
        """
        The base step method for the environment. It executes the action in the environment
        and returns the observation, reward, done flag, and additional information of the
        environment. Other functions is expected to be implemented by the subclass.

        Returns:
        - observation (dict): The observation of the environment
        - reward (float): The reward of the environment
        - done (bool): The done flag of the environment
        - info (dict): The information of the environment
        """
        return {}, 0, False, {}


class OfflineQLabEnv(BaseQLabEnv):
    """
    The OfflineQLabEnv class is a subclass of the BaseQLabEnv class, which is used in the
    offline RL training. It defines the methods for getting the dataset and normalizing
    the score.

    Attributes:
    - episode_steps (int): The number of steps in the episode
    - reference_max_score_per_step (float): The reference maximum score per step
    - reference_min_score_per_step (float): The reference minimum score per step
    """

    def __init__(
        self,
        reference_max_score_per_step: float,
        reference_min_score_per_step: float
    ) -> None:
        """
        Initialize the OfflineQLabEnv object with the reference maximum score per step

        Parameters:
        - reference_max_score_per_step (float): The reference maximum score per step
        - reference_min_score_per_step (float): The reference minimum score per step
        """
        self.episode_steps: int = 0
        self.reference_max_score_per_step: float = reference_max_score_per_step
        self.reference_min_score_per_step: float = reference_min_score_per_step

    def get_normalized_score(self, score: float) -> float:
        """
        Get the normalized score based on the reference maximum and minimum score per step.

        Parameters:
        - score (float): The score to be normalized

        Returns:
        - float: The normalized score
        """
        return (score - self.reference_min_score_per_step) / \
            (self.reference_max_score_per_step - self.reference_min_score_per_step)

    @abstractmethod
    def get_dataset(self, *args) -> None:
        """
        Get the dataset for the offline RL training. This method is expected to be
        implemented by the subclass.

        Returns:
        - None
        """
        ...

class OnlineQLabEnv(BaseQLabEnv):
    """
    OnlineQLabEnv is a subclass of the Base QLab Environment class, which is used in
    the online reinforcement learning training. It defines the methods for setting up
    the environment, handling the spawn position, and resetting the environment. It is
    expected to use the roadmaps provided by Quanser.

    Attributes:
    - dt (float): The time step of the environment
    - offsets (Tuple[float]): The offsets of the environment
    - privileged (bool): The privileged information flag
    - max_episode_steps (int): The maximum number of steps in the episode
    - simulator (QLabSimulator): The simulator of the environment
    """

    def __init__(
        self,
        simulator: QLabSimulator,
        roadmap: ACCRoadMap,
        dt: float = 0.05,
        privileged: bool = False,
    ) -> None:
        """
        Initialize the OnlineQLabEnv object with the time step, action size, offsets, and
        privileged information flag.

        Parameters:
        - simulator (QLabSimulator): The simulator of the environment
        - roadmap (RoadMap): The roadmap of the environment
        - dt (float): The time step of the environment
        - privileged (bool): The privileged information flag
        """
        self.dt: float = dt
        self.privileged: bool = privileged
        self.max_episode_steps: int = DEFAULT_MAX_STEPS

        self.roadmap: ACCRoadMap = roadmap
        self.simulator: QLabSimulator = simulator

    def reset(self) -> Tuple[dict, float, bool, dict]:
        """
        The base reset method for the environment. It resets the environment to the
        initial state. Other functions is expected to be implemented by the subclass.

        Returns:
        - observation (dict): The observation of the environment
        - reward (float): The reward of the environment
        - done (bool): The done flag of the environment
        - info (dict): The information of the environment
        """
        done: bool = False
        self.simulator.reset_map()
        observation, reward, info = self._init_step_params()

        self.episode_steps: int = 0
        return observation, reward, done, info

QLabEnvironment = OnlineQLabEnv
