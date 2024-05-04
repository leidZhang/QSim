import numpy as np
from typing import Dict, Tuple


class SequenceRolloutBuffer:
    def __init__(self, buffer_size: int, observation_shape: Tuple, action_dim: int):
        self.buffer_size = buffer_size
        self.obs_shape = observation_shape
        self.action_dim = action_dim
        self.pos = 0
        self.full = False

        self.reset()

    def __len__(self):
        return self.buffer_size if self.full else self.pos

    def reset(self):
        self.observations = np.zeros((self.buffer_size, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.next_states = np.zeros((self.buffer_size, *self.obs_shape), dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=bool)
        self.pos = 0
        self.full = False

    def add(self, state, action, reward, next_state, done):
        index = self.pos % self.buffer_size
        self.observations[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.dones[index] = done

        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True

    def store_transition(self, state, action, reward, next_state, done):
        self.add(state, action, reward, next_state, done)

    def sample(self, batch_size: int):
        max_ind = self.buffer_size if self.full else self.pos
        index = np.random.choice(max_ind, size=batch_size, replace=False)
        return {
            'states': self.observations[index],
            'actions': self.actions[index],
            'rewards': self.rewards[index],
            'next_states': self.next_states[index],
            'dones': self.dones[index]
        }

    # def _get_samples(
    #     self,
    #     batch_indices: np.ndarray
    # ):
    #     data = {
    #         self.observations[batch_inds],
    #         self.actions[batch_inds],
    #         self.values[batch_inds].flatten(),
    #         self.log_probs[batch_inds].flatten(),
    #         self.advantages[batch_inds].flatten(),
    #         self.returns[batch_inds].flatten(),
    #     }