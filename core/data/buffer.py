import numpy as np
from typing import Dict, Tuple
from core.utils.aggregation_utils import map_structure

class SequenceRolloutBuffer():
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        sequence_length: int,
        observation_shape: Tuple,
        action_dim: int = 2, 
        gae_lambda: float = 1.0,
        gamma: float = 0.99
    ):
        self.buffer_size = buffer_size
        self.sequence_length = sequence_length
        self.obs_shape = observation_shape
        self.action_dim = action_dim
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.pos = 0
        self.full = False

        self.reset()

    def reset(self):
        self.observations = np.zeros((self.buffer_size, self.sequence_length, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.sequence_length, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.sequence_length), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.sequence_length), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.sequence_length), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.sequence_length), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.sequence_length), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.sequence_length), dtype=np.float32)

        self.pos = 0
        self.full = True

    def add(
        self,
        obs,
        action,
        reward,
        episode_start,
        value,
        log_prob
    ):
        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()

        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True

    def get(
        self, 
        batch_size: int = 1
    ):
        indices = np.random.permutation(self.pos)

    def _get_samples(
        self,
        batch_indices: np.ndarray
    ):
        data = {
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        }