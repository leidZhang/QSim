import os
import glob
import random
from typing import List, Set, Tuple, Dict

import numpy as np

from core.environment import OfflineQLabEnv
from core.roadmap import ACCRoadMap
from constants import MAX_LOOKAHEAD_INDICES
from constants import GOAL_THRESHOLD, DEFAULT_MAX_STEPS
from core.policies.pt_policy import PTPolicy
from .vehicle import ReinformerPolicy


def unpack_task(state: np.ndarray) -> np.ndarray:
    task: np.ndarray = state[:-15]
    pointer: int = 0
    while task[pointer] > 0: # -99 is the place holder
        pointer += 1
    return task[:pointer+1]


def unpack_ego_state(state: np.ndarray) -> np.ndarray:
    return state[400:-15]


class OfflineWaypointEnv(OfflineQLabEnv):
    def __init__(self, reference_max_score: float, reference_min_score: float) -> None:
        super().__init__(reference_max_score, reference_min_score)
        self.roadmap: ACCRoadMap = ACCRoadMap()

    def _update_current_waypoint_index(self, ego_state: np.ndarray) -> None:
        orig: np.ndarray = ego_state[:2]
        local_waypoints: np.ndarray = np.roll(
            self.waypoints, -self.current_waypoint_index, axis=0
        )[:MAX_LOOKAHEAD_INDICES]
        norm_dist: np.ndarray = np.linalg.norm(local_waypoints - orig, axis=1)
        dist_ix: int = np.argmin(norm_dist)
        self.current_waypoint_index = (self.current_waypoint_index + dist_ix) % self.waypoints.shape[0]

    def get_dataset(self, folder_path: str, use_abs_path: bool = False) -> None:
        # get the dataset path
        if not use_abs_path:
            dataset_path: str = os.path.join(os.getcwd(), folder_path)
        else:
            dataset_path: str = folder_path
        # get the dataset list
        dataset_files: List[str] = glob.glob(f"{dataset_path}/*.npz")
        self.data_pool: Set[str] = set(dataset_files)

    def handle_reward(self, action: list, ego_state: np.ndarray, expert_action: np.ndarray) -> float:
        progress: int = self.current_waypoint_index - self.last_waypoint_index
        (reward := progress * 0.125) # create reward while calculating progress reward
        deviate: float = -abs(action[1] - expert_action[1]) # panelty for the deviation
        reward += deviate * progress * 0.104
        return reward

    def step(
        self,
        action: np.ndarray,
        metrics: np.ndarray,
        expert_action: np.ndarray
    ) -> Tuple[dict, float, bool, dict]:
        observation, reward, info = self._init_step_params()

        state: np.ndarray = self.replay_data["states"][self.episode_steps]
        ego_state: np.ndarray = unpack_ego_state(state) # x, y, yaw, v, ω, a
        self._update_current_waypoint_index(ego_state)
        reward += self.handle_reward(action=action, expert_action=expert_action)
        self.last_waypoint_index: int = self.current_waypoint_index

        done: bool = self.episode_steps == len(self.replay_data["states"]) - 1
        observation["waypoints"] = self.replay_data["observations"][self.episode_steps]
        observation["state"] = state
        self.episode_steps += 1

        return observation, reward, done, info

    def reset(self) -> Tuple[dict, float, bool, dict]:
        observation, reward, info = self._init_step_params()

        # prepare the evalutaion replay data
        episode_data_path: str = random.choice(tuple(self.data_pool))
        self.data_pool.remove(episode_data_path)
        self.replay_data: dict = np.load(episode_data_path, allow_pickle=True)

        # reset variables for replay the steps
        self.episode_steps: int = 0
        self.current_waypoint_index: int = 0
        self.last_waypoint_index: int = self.current_waypoint_index
        state = self.replay_data["states"][self.episode_steps]
        task: np.ndarray = unpack_task(state)
        self.waypoints: np.ndarray = self.roadmap.generate_path(task)
        # prepare the observation dict
        observation["waypoints"] = self.replay_data["observations"][self.episode_steps]
        observation["state"] = state

        return observation, reward, False, info
