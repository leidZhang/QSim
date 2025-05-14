from typing import Tuple, Any
import time

import numpy as np
import torch

from core.policies.pt_policy import PTPolicy
from settings import *

class ReinformerPolicy(PTPolicy):
    def reset(
        self, 
        eval_batch_size: int=1,
        max_test_ep_len: int=100000,
        context_len: int=CONTEXT_LEN,
        state_mean: float=STATE_MEAN,
        state_std: float=STATE_STD,
        state_dim: int=STATE_DIM,
        act_dim: int=ACT_DIM,
        device: str=DEVICE
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

    def handle_observation(self, observation: dict) -> dict:
        task = observation['task']
        waypoints = observation['waypoints'].reshape(-1)
        state = observation['state_info']
        observation['waypoints'] = np.concatenate([waypoints, state, task])

        return observation

    def execute(self, observation: dict) -> Tuple[np.ndarray, dict]:
        observation = self.handle_observation(observation)

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
    
    step = execute