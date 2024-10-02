from typing import *

import torch
import numpy as np

from core.templates import PolicyAdapter
from core.policies.pure_persuit import PurePursuitPolicy
from .agent import DQNPolicy


class CompositeDQNPolicy(PolicyAdapter):
    def __init__(
        self, 
        max_lookahead_distance: float = 0.70,
        action_size: float = 1
    ) -> None:
        self.steering_control: PurePursuitPolicy = PurePursuitPolicy(max_lookahead_distance)
        self.throttle_control: DQNPolicy = DQNPolicy(action_size=action_size)

    def __handle_throttle(self, observation: Dict[str, Any]) -> float:
        state: Dict[str, torch.Tensor] = self.throttle_control.process_observation(observation)
        throttle: float = self.throttle_control.select_action(state)
        return throttle
    
    def __handle_steering(self, observation: Dict[str, Any]) -> float:
        action, _ = self.steering_control(observation)
        steering: float = action[1]
        return steering

    def execute(self, observation: Dict[str, Any]) -> Tuple[np.ndarray, dict]:
        steering: float = self.__handle_steering(observation)
        throttle: float = self.__handle_throttle(observation)
        action: np.ndarray = np.array([throttle, steering])
        return action, {}

    def reset(self) -> None:
        # Read the existing weight file and update the weights
        raise NotImplementedError("Not implemented yet")
        ...