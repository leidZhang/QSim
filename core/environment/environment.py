import math
from abc import abstractmethod
from typing import Tuple, Dict

from gym import Env
import numpy as np

from .simulator import QLabSimulator
from constants import DEFAULT_MAX_STEPS

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QLabEnvironment(Env):
    def __init__(self, dt: float = 0.05, action_size: int = 2, privileged: bool = False) -> None:
        self.action_size: int = action_size
        self.privileged: bool = privileged
        self.max_episode_steps: int = DEFAULT_MAX_STEPS
        self.simulator: QLabSimulator = QLabSimulator(dt)

    def setup(self, nodes: Dict[str, np.ndarray], sequence: np.ndarray) -> None:
        self.simulator.render_map()
        self.set_waypoint_sequence(sequence)
        self.set_nodes(nodes=nodes)

    def set_nodes(self, nodes: Dict[str, np.ndarray]) -> None:
        self.nodes: Dict[str, np.ndarray] = nodes

    def set_waypoint_sequence(self, sequence: np.ndarray) -> None:
        self.waypoint_sequence: np.ndarray = sequence
        self.goal: np.ndarray = self.waypoint_sequence[-1]

    def init_step_params(self) -> tuple:
        observation: dict = {}
        reward: float = 0.0
        info: dict = {}
        return observation, reward, info

    def cal_waypoint_angle(self, delta_x: float, delta_y: float) -> float:
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
        if waypoint_index < 0 or waypoint_index >= len(self.waypoint_sequence):
            raise ValueError('Invalid Waypoint index format')

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
        if node_index not in self.nodes.keys():
            raise ValueError("Index not exist!")
        node_pose: np.ndarray = self.nodes[node_index]
        x_position, y_position, orientation = node_pose
        return [x_position, y_position, 0], [0, 0, orientation]

    def reset(self, location, orientation) -> Tuple[dict, float, bool, dict]:
        self.simulator.reset_map(location=location, orientation=orientation)
        done: bool = False
        observation, reward, info = self.init_step_params()
        self.deviate_steps: int = 0
        self.episode_steps: int = 0
        return observation, reward, done, info

    @abstractmethod
    def handle_spawn_pos(self, *args) -> Tuple[list, list]:
        ...

    @abstractmethod
    def handle_reward(self, *args) -> Tuple[float, bool]:
        ...

    @abstractmethod
    def step(self, action: np.ndarray, metrics: np.ndarray, *args) -> Tuple[dict, float, bool, dict]:
        ...
