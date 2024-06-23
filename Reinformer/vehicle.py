from typing import Tuple, Any
import time

import numpy as np
import torch

from core.utils.performance import elapsed_time
from core.policies.pt_policy import PTPolicy
from qvl.qlabs import QuanserInteractiveLabs
from td3.vehicle import WaypointCar


class ReinformerPolicy(PTPolicy):
    def setup(
        self, 
        eval_batch_size: int,
        max_test_ep_len: int,
        context_len: int,
        state_mean: float,
        state_std: float,
        state_dim: int,
        act_dim: int,
        device: str
    ) -> None:
        self.device: str = device
        self.act_dim: int = act_dim
        self.context_len: int = context_len
        self.step_counter: int = 0
        self.timesteps: torch.Tensor = torch.arange(start=0, end=max_test_ep_len, step=1)
        self.timesteps = self.timesteps.repeat(eval_batch_size, 1).to(device)
        self.state_mean: torch.Tensor = torch.from_numpy(state_mean).to(device)
        self.state_std: torch.Tensor = torch.from_numpy(state_std).to(device)
        self.actions: torch.Tensor = torch.zeros(
            (eval_batch_size, max_test_ep_len, act_dim),
            dtype=torch.float32,
            device=device,
        )
        self.states: torch.Tensor = torch.zeros(
            (eval_batch_size, max_test_ep_len, state_dim),
            dtype=torch.float32,
            device=device,
        )
        self.returns_to_go: torch.Tensor = torch.zeros(
            (eval_batch_size, max_test_ep_len, 1),
            dtype=torch.float32,
            device=device,
        )

    def execute(self, observation: dict) -> Tuple[np.ndarray, dict]:
        observation_shape = self.states[0, self.step_counter].shape
        self.states[0, self.step_counter] = torch.from_numpy(observation['waypoints'].reshape(observation_shape)).to(self.device)
        self.states[0, self.step_counter] = (self.states[0, self.step_counter] - self.state_mean) / self.state_std
        if self.step_counter < self.context_len:
            _, action_predict, _ = self.model.forward(
                self.timesteps[:, :self.context_len],
                self.states[:, :self.context_len],
                self.actions[:, :self.context_len],
                self.returns_to_go[:, :self.context_len],
            )
        else:
            _, action_predict, _ = self.model.forward(
                self.timesteps[:, self.step_counter - self.context_len + 1 : self.step_counter + 1],
                self.states[:, self.step_counter - self.context_len + 1 : self.step_counter + 1],
                self.actions[:, self.step_counter - self.context_len + 1 : self.step_counter + 1],
                self.returns_to_go[:, self.step_counter - self.context_len + 1 : self.step_counter + 1],
            )
        action: torch.Tensor = action_predict.mean.reshape(1, -1, self.act_dim)[0, -1].detach()
        self.step_counter += 1
        return action.cpu().numpy(), {}


class ReinformerCar(WaypointCar):
    def setup(
        self, 
        waypoints: np.ndarray, 
        init_waypoint_index: int, 
        policy: ReinformerPolicy
    ) -> None:
        if policy is None:
            raise ValueError("Policy cannot be None")
        super().setup(waypoints, init_waypoint_index)
        self.policy: ReinformerPolicy = policy

    def handle_observation(self, orig: np.ndarray, rot: np.ndarray) -> None:
        super().handle_observation(orig, rot)
        far_index: int = (self.current_waypoint_index + 49) % self.waypoints.shape[0]
        self.observation['state'] = np.concatenate((
            self.ego_state, # ego state
            self.waypoints[self.current_waypoint_index], # closest waypoint
            self.waypoints[far_index] # far waypoint
        )) # state info

    def execute(self) -> None:
        # get the start time of the loop
        start_time: float = time.time()
        # get the action from the policy
        action, _ = self.policy.execute(self.observation)
        # apply the action to the vehicle
        self.running_gear.read_write_std(action[0] * 0.08, action[1])
        self.update_state()
        # sleep for the remaining time
        execute_time: float = elapsed_time(start_time)
        sleep_time: float = self.monitor.dt - execute_time
        time.sleep(max(0, sleep_time))