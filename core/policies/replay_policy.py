
from typing import Union, Tuple
import numpy as np

from ..templates.base_policy import BasePolicy, PolicyAdapter


class ReplayPolicy(BasePolicy):
    def __init__(
        self,
        replay_data: dict,
        policy_for_replay: Union[BasePolicy, PolicyAdapter]
    ) -> None:
        self.data: dict = replay_data # get from npz files, pkl files, etc.
        self.policy_for_replay: Union[BasePolicy, PolicyAdapter] = \
            policy_for_replay # the policy to replay the data
        # determine the execute method based on the policy_for_replay


    def _replay_action(self, step_index: int, callback) -> Tuple[dict, dict]:
        return callback(self.data, step_index), {} # directly return the action

    def _replay_observation(self, step_index: int, callback) -> Tuple[dict, dict]:
        obs: dict = callback(self.data, step_index) # get the observation
        action, metric = self.policy_for_replay.execute(obs)
        return action, metric

    def execute(self, step_index: int, callback) -> Tuple[dict, dict]:
        if self.policy_for_replay is None:
            return self._replay_action(step_index, callback)
        return self._replay_observation(step_index, callback)
