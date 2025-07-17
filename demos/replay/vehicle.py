import time
from typing import Tuple

import numpy as np

from core.qcar import PhysicalCar
from core.policies.base_policy import BasePolicy
from core.policies.replay_policy import ReplayPolicy


class ReplayCar(PhysicalCar):
    def __init__(
        self, 
        replay_data: dict,
        throttle_coeff: float = 0.3, 
        steering_coeff: float = 0.5,
        policy_for_replay: BasePolicy = None
    ) -> None:
        super().__init__(throttle_coeff, steering_coeff)
        self.replay_policy = ReplayPolicy(
            replay_data=replay_data, 
            policy_for_replay=policy_for_replay
        )
        if policy_for_replay is None:
            self.execution_method = self._get_action
        else:
            self.execution_method = self._get_observation

    def _get_observation(self, replay_data: dict, start_index: int) -> dict:
        waypoints: np.ndarray = replay_data["waypoints"][start_index]
        return {"waypoints": waypoints}
    
    def _get_action(self, replay_data: dict, start_index: int) -> Tuple:
        action: np.ndarray = replay_data["action"][start_index]
        return action

    def execute(self, step_index: int) -> Tuple:
        start_time = time.time()
        action, _ = self.replay_policy.execute(step_index, self.execution_method)
        
        self.running_gear.read_write_std(
            throttle=action[0] * 0.08, 
            steering=action[1] * self.steering_coeff,
        )
        # Usually we expect that we have the time difference betweent the 2 steps
        sleep_time = 0.0135 - (time.time() - start_time)
        time.sleep(max(0, sleep_time))

