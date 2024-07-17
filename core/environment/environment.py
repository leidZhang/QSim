import math
import warnings
from abc import abstractmethod
from typing import Tuple, Dict

from gym import Env
import numpy as np

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

    @abstractmethod
    def handle_reward(self, *args) -> Tuple[float, bool]:
        """
        This method handles the reward calculation for the environment.
        The function should return the reward and the done flag, and is 
        expected to be implemented by the subclass.
        """
        ...

    @abstractmethod
    def step(self, action: np.ndarray, metrics: np.ndarray, *args) -> Tuple[dict, float, bool, dict]:
        """
        This method executes the action in the environment and returns the observation, 
        reward, done flag, and additional information of the environment.
        """
        ...


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
    - action_size (int): The size of the action space
    - offsets (Tuple[float]): The offsets of the environment
    - privileged (bool): The privileged information flag
    - max_episode_steps (int): The maximum number of steps in the episode
    - simulator (QLabSimulator): The simulator of the environment
    """

    def __init__(
        self,
        dt: float = 0.05,
        action_size: int = 2,
        privileged: bool = False,
        offsets: Tuple[float] = (0, 0)
    ) -> None:
        """
        Initialize the OnlineQLabEnv object with the time step, action size, offsets, and
        privileged information flag.

        Parameters:
        - dt (float): The time step of the environment
        - action_size (int): The size of the action space
        - offsets (Tuple[float]): The offsets of the environment
        - privileged (bool): The privileged information flag
        """
        self.dt: float = dt
        self.action_size: int = action_size
        self.offsets: Tuple[float] = offsets
        self.privileged: bool = privileged
        self.max_episode_steps: int = DEFAULT_MAX_STEPS
        self.simulator: QLabSimulator = QLabSimulator(offsets=self.offsets)

    def setup(self, nodes: Dict[str, np.ndarray], sequence: np.ndarray) -> None:
        """
        Setup the environment with the nodes and waypoint sequence.

        Parameters:
        - nodes (Dict[str, np.ndarray]): The nodes of the environment
        - sequence (np.ndarray): The waypoint sequence of the environment
        """
        self.simulator.render_map()
        self.set_waypoint_sequence(sequence)
        self.set_nodes(nodes=nodes)

    def set_nodes(self, nodes: Dict[str, np.ndarray]) -> None:
        """
        Set the key nodes of the environment.

        Parameters:
        - nodes (Dict[str, np.ndarray]): The nodes of the environment

        Returns:
        - None
        """
        self.nodes: Dict[str, np.ndarray] = nodes

    def set_waypoint_sequence(self, sequence: np.ndarray) -> None:
        """
        Set the waypoint sequence of the environment.

        Parameters:
        - sequence (np.ndarray): The waypoint sequence of the environment

        Returns:
        - None
        """
        self.waypoint_sequence: np.ndarray = sequence
        self.goal: np.ndarray = self.waypoint_sequence[-1]

    def recover_state_info(self, state: np.ndarray, recover_indices: list) -> np.ndarray:
        """
        This method recovers the state information based on the recover indices.

        Parameters:
        - state (np.ndarray): The state information
        - recover_indices (list): The recover indices

        Returns:
        - np.ndarray: The recovered state information
        """
        warnings.warn("This method is no longer maintained.")
        recovered_state: np.ndarray = state.copy()
        for index in recover_indices:
            recovered_state[index] -= self.offsets[0]
            recovered_state[index + 1] -= self.offsets[1]
        return recovered_state

    def cal_waypoint_angle(self, delta_x: float, delta_y: float) -> float:
        """
        Calculate the angle of the waypoint based on the delta x and delta y.

        Parameters:
        - delta_x (float): The delta x of the waypoint
        - delta_y (float): The delta y of the waypoint

        Returns:
        - float: The angle of the waypoint
        """
        if delta_x < 0 and delta_y == 0:
            return math.pi # up to bottom
        elif delta_x == 0 and delta_y > 0:
            return math.pi / 2 # right to left
        elif delta_x < 0 and delta_y > 0:
            return math.atan(delta_y / delta_x) + math.pi # right bottom
        elif delta_x > 0 and delta_y > 0:
            return math.atan(delta_y / delta_x) # left bottom
        elif delta_x > 0 and delta_y == 0:
            return 0 # bottom to up
        elif delta_x > 0 and delta_y < 0:
            return math.atan(delta_y / delta_x) # left top
        elif delta_x == 0 and delta_y < 0:
            return 3 * math.pi / 2 # left to right
        else:
            return math.atan(delta_y / delta_x) + math.pi # right top

    def spawn_on_waypoints(self, waypoint_index: int = 0) -> Tuple[list, list]:
        """
        Spawn the vehicle on the waypoints based on the waypoint index.

        Parameters:
        - waypoint_index (int): The index of the waypoint

        Returns:
        - Tuple[list, list]: The position and orientation of the vehicle on the designated waypoint
        """
        if waypoint_index < 0 or waypoint_index >= len(self.waypoint_sequence):
            raise ValueError('Invalid Waypoint index format!')

        # handle final index
        if waypoint_index < len(self.waypoint_sequence) - 1:
            current_waypoint: np.ndarray = self.waypoint_sequence[waypoint_index]
            next_waypoint: np.ndarray = self.waypoint_sequence[waypoint_index+1]
        else:
            current_waypoint: np.ndarray = self.waypoint_sequence[waypoint_index-1]
            next_waypoint: np.ndarray = self.waypoint_sequence[waypoint_index]
        # calculate x, y coordinates
        x_position: float = current_waypoint[0]
        y_position: float = current_waypoint[1]
        # calcualte angle
        delta_x: float = next_waypoint[0] - current_waypoint[0]
        delta_y: float = next_waypoint[1] - current_waypoint[1]
        orientation: float = self.cal_waypoint_angle(delta_x, delta_y)

        return [x_position, y_position, 0], [0, 0, orientation]

    def spawn_on_nodes(self, node_index: int) -> Tuple[list, list]:
        """
        Spawn the vehicle on the nodes based on the node index.

        Parameters:
        - node_index (int): The index of the node

        Returns:
        - Tuple[list, list]: The position and orientation of the vehicle on the designated node
        """
        if node_index not in self.nodes.keys():
            raise ValueError("Index does not exist!")
        node_pose: np.ndarray = self.nodes[node_index]
        x_position, y_position, orientation = node_pose
        return [x_position, y_position, 0], [0, 0, orientation]

    def reset(self, location, orientation) -> Tuple[dict, float, bool, dict]:
        """
        The base reset method for the environment. It resets the environment to the
        initial state. Other functions is expected to be implemented by the subclass.
        """
        self.simulator.reset_map(location=location, orientation=orientation)
        done: bool = False
        observation, reward, info = self._init_step_params()
        self.deviate_steps: int = 0
        self.episode_steps: int = 0
        return observation, reward, done, info

    def handle_spawn_pos(self, *args) -> Tuple[list, list]:
        """
        Handle the spawn position of the vehicle in the environment. The function is
        expected to be implemented by the subclass.
        """
        ...

QLabEnvironment = OnlineQLabEnv
