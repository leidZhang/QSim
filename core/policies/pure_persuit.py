from typing import Tuple

import torch
import numpy as np
import random

#project imports
from core.models.torch.model import Model
from core.data.preprocessor import Preprocessor
from core.policies.network import NetworkPolicy
from core.utils.agg_utils import map_structure
from core.policies.base_policy import PolicyAdapter
from constants import action_v

class PurePursuitPolicy:
    def __init__(self, max_lookahead_distance: float = 0.5) -> None:
        self.max_lookahead_distance = max_lookahead_distance

    def __call__(self, obs: dict) -> Tuple[dict, dict]:
        # action = np.array([0.074, 0.0]) #v, steer
        action: np.ndarray = np.array([1, 0.0])  # v, steer
        metrics: dict = {}

        state: np.ndarray = np.zeros((6,), dtype=np.float32) #obs["state"]
        waypoints: np.ndarray = obs["waypoints"]
        metrics["waypoints"] = waypoints

        lad: float = 0.0
        i: int = 0
        for i in range(waypoints.shape[0] - 1):
            current_waypoint_x: float = waypoints[i, 0]
            current_waypoint_y: float = waypoints[i, 1]
            next_waypoint_x: float = waypoints[i + 1, 0]
            next_waypoint_y: float = waypoints[i + 1, 1]

            lad = lad + np.hypot(next_waypoint_x - current_waypoint_x, next_waypoint_y - current_waypoint_y)
            if lad > self.max_lookahead_distance:
                break

        tx, ty = waypoints[i]

        # compute steer action
        x, y, yaw = state[:3]        
        alpha: float = np.arctan2(ty - y, tx - x) - yaw
        l: float = np.sqrt((x - tx)**2 + (y - ty)**2)
        theta: float = np.arctan2(2 * 0.256 * np.sin(alpha), l)
        action[1] = theta / 0.5

        return action, metrics
    

class PurePursuiteAdaptor(PolicyAdapter):
    def __init__(self, max_lookahead_distance=0.5) -> None:
        self.policy = PurePursuitPolicy(max_lookahead_distance)

    def execute(self, obs) -> Tuple[dict, dict]:
        return self.policy(obs)


class PurePursuitNetworkPolicy(NetworkPolicy):
    def __init__(self, device, max_lookahead_distance=0.5):
        self.max_lookahead_distance = max_lookahead_distance

        self.device = device
        self.model = Model(device=device)

        self.preprocessor = Preprocessor(
            image_key="image"
        )
        self.recurrent_state = None

        self.mins = np.array([-0.05])
        self.maxs = np.array([0.2])

    def reset_state(self):
        self.recurrent_state = self.model.init_state()

    def __call__(self, obs):
        action = np.array([0.08, 0.0]) #v, steer
        metrics = {}

        batch = self.preprocessor.apply(obs, expand_tb=True)
        obs_model: Dict[str, torch.Tensor] = map_structure(map_structure(batch, torch.from_numpy), lambda x: x.to(self.device))  # type: ignore

        waypoints, recurrent_state, metrics = self.model.inference(obs_model, self.recurrent_state)
        self.recurrent_state = recurrent_state
        waypoints = waypoints.detach().cpu().numpy()
        metrics["waypoints"] = waypoints

        if np.random.rand() < 0.01:
            print("WAYPOINTS")
            print(waypoints[:10])
            print(obs["waypoints"][:10])

        '''act, recurrent_state = self.model.inference(obs_model, self.recurrent_state)
        self.recurrent_state = recurrent_state

        action_noise = (0.9999**steps) * np.random.normal(loc=np.array([0.05]), scale=np.array([0.2]))
        action[0] = np.clip(np.clip(act, self.mins, self.maxs) + action_noise, self.mins, self.maxs)[0]'''

        state = np.zeros((6,)) #obs["state"]
        #waypoints = obs["waypoints"]

        lad = 0.0
        i = 0
        for i in range(waypoints.shape[0] - 1):
            current_waypoint_x = waypoints[i, 0]
            current_waypoint_y = waypoints[i, 1]
            next_waypoint_x = waypoints[i + 1, 0]
            next_waypoint_y = waypoints[i + 1, 1]

            lad = lad + np.hypot(next_waypoint_x - current_waypoint_x, next_waypoint_y - current_waypoint_y)
            if lad > self.max_lookahead_distance:
                break

        tx, ty = waypoints[i]

        #compute steer action
        x, y, yaw = state[:3]
        alpha = np.arctan2(ty - y, tx - x) - yaw
        l = np.sqrt((x - tx)**2 + (y - ty)**2)
        theta = np.arctan2(2 * 0.256 * np.sin(alpha), l)
        action[1] = theta

        # # add noise
        # with torch.no_grad():
        #     action_yaw = torch.from_numpy(np.array(theta))
        #     epsilon = max(1 - data_size / 400_000, 0.04)
        #     # epsilon = 0
        #     rand_action_yaw = torch.rand(action_yaw.shape)
        #     rand_action_yaw = rand_action_yaw * 2 - 1
        #     if random.uniform(0, 1) < epsilon:
        #         action_yaw = rand_action_yaw
        #
        #     action_yaw = action_yaw.cpu().data.numpy().flatten()

        return action, metrics