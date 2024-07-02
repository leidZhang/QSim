import os
import glob
import random
from typing import List, Set, Tuple

import numpy as np
from torch.nn import Module

from core.environment import OfflineQLabEnv
from core.policies import PolicyAdapter, PurePursuiteAdaptor
from core.roadmap import ACCRoadMap
from constants import MAX_LOOKAHEAD_INDICES
from core.policies.pt_policy import PTPolicy
from .vehicle import ReinformerPolicy
from .settings import *


def reinformer_qcar_eval(
    model: Module,
    device: str,
    context_len: int,
    env: OfflineQLabEnv,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    num_eval_ep: int = 10,
    max_test_ep_len: int = 1000
) -> Tuple[float, float, float, float]:
    rewards: List[float] = []
    lengths: List[float] = []

    for _ in range(num_eval_ep):
        # init agent and expert
        expert: PolicyAdapter = PurePursuiteAdaptor(max_lookahead_distance=0.5)
        agent: PTPolicy = ReinformerPolicy(model=model, weight_path=None) # use model in the trainer
        agent.setup(
            eval_batch_size=1,
            max_test_ep_len=max_test_ep_len,
            context_len=context_len,
            state_mean=state_mean, # how to get this when deploy on a real car?
            state_std=state_std, # how to get this when deploy on a real car?
            state_dim=STATE_DIM,
            act_dim=ACT_DIM,
            device=device,
        )

        observation, episode_reward, done, _ = env.reset() # state, reward, done, info
        while not done:
            # generate the action
            action, metrics = agent.execute(observation=observation)
            expert_action, _ = expert.execute() * 0.5
            # apply actions to the reward
            observation, reward, done, _ = env.step(action, metrics, expert_action)
            episode_reward += reward
        # append the reward to the history list
        rewards.append(episode_reward / env.episode_steps) # average reward per step
        lengths.append(env.episode_steps) # what does this used for?

    return np.array(rewards).mean(), np.array(rewards).std(), np.array(lengths).mean(), np.array(lengths).std()


def unpack_task(state: np.ndarray) -> np.ndarray:
    task: np.ndarray = state[:-15]
    pointer: int = 0
    while task[pointer] > 0: # -99 is the place holder
        pointer += 1
    return task[:pointer+1]


def unpack_ego_state(state: np.ndarray) -> np.ndarray:
    return state[400:-15]


class OfflineWaypointEnv(OfflineQLabEnv):
    def __init__(
        self,
        reference_max_score_per_step: float,
        reference_min_score_per_step: float
    ) -> None:
        super().__init__(reference_max_score_per_step, reference_min_score_per_step)
        self.roadmap: ACCRoadMap = ACCRoadMap()

    def _update_current_waypoint_index(self, ego_state: np.ndarray) -> None:
        orig: np.ndarray = ego_state[:2]
        local_waypoints: np.ndarray = np.roll(
            self.waypoints, -self.current_waypoint_index, axis=0
        )[:MAX_LOOKAHEAD_INDICES]
        norm_dist: np.ndarray = np.linalg.norm(local_waypoints - orig, axis=1)
        dist_ix: int = np.argmin(norm_dist)
        self.current_waypoint_index = (self.current_waypoint_index + dist_ix) % self.waypoints.shape[0]

    def _prepare_episode_data(self) -> None:
        episode_data_path: str = random.choice(tuple(self.data_pool))
        self.data_pool.remove(episode_data_path)
        self.replay_data = np.load(episode_data_path, allow_pickle=True)
        self.max_episode_length: int = len(self.replay_data["states"])

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

        done: bool = self.episode_steps == self.max_episode_length - 1
        observation["waypoints"] = self.replay_data["observations"][self.episode_steps]
        observation["state"] = state
        self.episode_steps += 1

        return observation, reward, done, info

    def reset(self) -> Tuple[dict, float, bool, dict]:
        observation, reward, info = self._init_step_params()
        self.replay_data: dict = {"state": []} # clear replay data dict
        self._prepare_episode_data()

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
