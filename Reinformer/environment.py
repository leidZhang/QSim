import random
from typing import Tuple, List

import torch
import numpy as np
import torch.nn as nn
import numpy as np
from gym import Env

from qvl.qlabs import QuanserInteractiveLabs

from core.environment import QLabEnvironment
from core.environment.detector import EpisodeMonitor
from constants import GOAL_THRESHOLD
from td3.vehicle import WaypointCar
from core.policies.pt_policy import PTPolicy
from core.roadmap.dispatcher import SingleTaskGenerator
from core.policies.pure_persuit import PurePursuiteAdaptor, PolicyAdapter
from .vehicle import ReinformerPolicy
from .settings import ACT_DIM, STATE_DIM


# TODO: Solve ong parameter list in the this function
def reinformer_car_eval(
    model: nn.Module,
    model_path: str,
    device: str,
    env: Env,
    context_len: int,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    num_eval_ep: int = 10,
    max_test_ep_len: int = 1000,
) -> Tuple[List[float], List[float]]:
    # initialize the agent and the expert
    agent: PTPolicy = ReinformerPolicy(model=model, model_path=model_path)
    expert: PolicyAdapter = PurePursuiteAdaptor()
    task_assigner: SingleTaskGenerator = SingleTaskGenerator()

    # initialize returns
    returns, lengths = [], []
    for _ in range(num_eval_ep):
        # initialize the task assigner
        start_index: int = random.randint(0, 23)
        task, waypoints = task_assigner.get_task(start_index)
        # Reinitialize the environment
        state, reward, done, _ = env.reset(task, waypoints)
        episode_return: float = 0
        episode_length: float = 0
        # setup the agent
        agent.setup(
            eval_batch_size=1,
            max_test_ep_len=max_test_ep_len,
            context_len=context_len,
            state_mean=state_mean,
            state_std=state_std,
            state_dim=STATE_DIM,
            act_dim=ACT_DIM,
            device=device
        )

        # execute the steps
        for _ in range(max_test_ep_len):
            print(state.keys())
            expert_action, _ = expert.execute(state)
            agent_action, _ = agent.execute(state)
            state, reward, done, _ = env.step(agent_action, expert_action)
            episode_return += reward
            episode_length += 1
            if done:
                returns.append(episode_return)
                lengths.append(episode_length)
                break

    return returns, lengths


class ReinformerQLabEnv(QLabEnvironment):
    def __init__(
        self,
        dt: float = 0.05,
        action_size: int = 2,
        privileged: bool = False,
        offsets: Tuple[float] = (0, 0)
    ) -> None:
        super().__init__(dt, action_size, privileged, offsets)
        self.simulator.render_map()

    def handle_reward(
        self,
        action: list,
        norm_dist: np.ndarray,
        ego_state: np.ndarray,
        dist_ix: float,
        expert_action: np.ndarray
    ) -> Tuple[float, bool]:
        # detect the anomalous episodes
        self.detector(action=action, orig=ego_state[:2])
        done: bool = False
        reward: float = 0.0

        # calculate the reward based on passed waypoints
        index: int = self.vehicle.current_waypoint_index
        progress: int = index - self.pre_index
        reward += progress * 0.125
        reward += -abs(action[1] - expert_action[1] / 2) * progress * 0.104
        self.pre_index = index

        # Max boundary
        if norm_dist[dist_ix] >= 0.80:
            done = True
            print(f"Max boundary reached!")
            self.vehicle.halt()  # stop the car

        if np.linalg.norm(self.goal - ego_state[:2]) < GOAL_THRESHOLD:
            done = True  # stop episode after this step
            print(f"Reached the goal!")
            self.vehicle.halt()  # stop the car

        return reward, done

    def step(
        self,
        action: np.ndarray,
        expert_action: np.ndarray
    ) -> Tuple[dict, float, bool, dict]:
        observation, reward, info = self.init_step_params()
        action: np.ndarray = self.vehicle.execute(action)

        ego_state: np.ndarray = self.vehicle.ego_state
        norm_dist: np.ndarray = self.vehicle.norm_dist
        dist_ix: int = self.vehicle.dist_ix
        reward, done = self.handle_reward(
            action, norm_dist, ego_state, dist_ix, expert_action
        )

        ego_state = self.vehicle.ego_state
        local_waypoints: np.ndarray = self.vehicle.observation['waypoints']
        observation['waypoints'] = local_waypoints
        observation['state'] = np.concatenate(
            [local_waypoints.reshape(-1), ego_state, self.task]
        )

        return observation, reward, done, info

    def reset(
        self,
        task: list,
        wayponts: np.ndarray,
        start_index: int = 0
    ) -> Tuple[dict, float, bool, dict]:
        # prepare the qcar spawn position
        self.waypoint_sequence: np.ndarray = wayponts
        self.goal: np.ndarray = wayponts[-1]
        location, orientation = self.spawn_on_waypoints(waypoint_index=start_index)
        observation, reward, done, info = super().reset(location, orientation)

        # prepare the qcar actor
        qlabs: QuanserInteractiveLabs = self.simulator.qlabs
        self.vehicle: WaypointCar = WaypointCar(
            actor_id=0,
            dt=0.0165,
            qlabs=qlabs,
            throttle_coeff=1,
            steering_coeff=1
        )
        self.vehicle.setup(self.waypoint_sequence, start_index)
        self.pre_index: int = start_index

        # prepare the observations
        self.task: np.ndarray = np.array(task)
        place_holder: np.ndarray = np.full(15 - len(task), -99)
        self.task = np.concatenate([self.task, place_holder])
        ego_state: np.ndarray = self.vehicle.ego_state
        local_waypoints: np.ndarray = self.vehicle.observation['waypoints']
        observation['waypoints'] = local_waypoints
        observation['state'] = np.concatenate(
            [local_waypoints.reshape(-1), ego_state, self.task]
        )

        # init fault tolerance
        self.detector: EpisodeMonitor = EpisodeMonitor(start_orig=location)
        return observation, reward, done, info
