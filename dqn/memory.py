import random
import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class SumTree:
    """
    SumTree data structure for Prioritized Experience Replay.
    """
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity  # Number of leaf nodes (transitions)
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def _propagate(self, idx, change):
        """
        Update tree nodes.
        """
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """
        Find sample on leaf node.
        """
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """
        Get total priority score.
        """
        return self.tree[0]

    def add(self, p, data):
        """
        Add new data with priority p.
        """
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0  # Overwrite when exceeding capacity

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        """
        Update priority score p at index idx.
        """
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        """
        Get data sample by priority.
        """
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayMemory:
    """
    Replay Memory with Prioritized Experience Replay.
    """
    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def push(self, *args):
        """
        Save a transition with maximum priority.
        """
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = 1e-5
        self.tree.add(max_p, Transition(*args))

    def sample(self, batch_size, seq_len):
        """
        Sample a batch of transitions.
        """
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -1)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, p):
        """
        Update the priority score of a transition.
        """
        self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries
