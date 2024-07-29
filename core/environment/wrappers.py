from typing import Union
from multiprocessing import Queue

import numpy as np
from gym import Wrapper

from .environment import QLabEnvironment


class ActionRewardResetWrapper(Wrapper):
    def __init__(self, env: QLabEnvironment, nodes: list, waypoints) -> None:
        super().__init__(env)
        self.action_size: int = env.action_size
        self.env.setup(nodes, waypoints)

    def step(self, action: np.ndarray, metrics: np.ndarray) -> tuple:
        observation, reward, done, info = self.env.step(action=action, metrics=metrics)
        observation["action"] = np.array(action, dtype=np.float32)
        observation["reward"] = np.array(reward, dtype=np.float32)
        observation["terminal"] = np.array(done, dtype=bool)
        observation["reset"] = np.array(False, dtype=bool)

        return observation, reward, done, info

    def reset(self) -> tuple:
        observation, reward, done, info = self.env.reset()
        observation["action"] = np.zeros((self.action_size, ), dtype=np.float32)
        observation["reward"] = np.array(0.0, dtype=np.float32)
        observation["terminal"] = np.array(False, dtype=bool)
        observation["reset"] = np.array(True, dtype=bool)

        return observation, reward, done, info


class CollectionWrapper(Wrapper):
    def __init__(self, env: QLabEnvironment, paddings: dict = {}) -> None:
        super().__init__(env)
        self.paddings: dict = paddings
        self.episode: list = []

    def step(self, action: np.ndarray, metrics: np.ndarray) -> tuple:
        observation, reward, done, info = self.env.step(action=action, metrics=metrics)
        self.episode.append(observation.copy())  # copy obs dict as a item and add it to self.episode list
        if not done:
            return observation, reward, done, info

        episode: dict = {}
        counter: dict = {}

        for k in self.episode[0]:  # k: item (state_info, image, waypoints, action, reward, terminal, reset)
            counter[k] = 0
            # print(f'k: {k}')
            data: list = []
            for t in self.episode:  # t: step index
                # print(f't: {t}')
                if k in self.paddings:
                    shape, value = self.paddings[k]
                    data.append(np.pad(
                        t[k], ((0, shape[0] - t[k].shape[0]), (0, 0)),
                        mode='constant', constant_values=(value, )
                    ))
                else:
                    # print(f"T[k] {t[k]}")
                    if t[k] is None:
                        counter[k] += 1
                    if t[k] is not None:
                        data.append(t[k])
            # print(f'counter of {k}: {counter[k]}')
            # print(f'len of {k}: {len(data)}')
            # print("Data type:", type(data))
            # print("Data: ", data)
            # print("Data shape:", np.array(data).shape)

            episode[k] = np.array(data)

        info["episode"] = episode
        return observation, reward, done, info

    def reset(self) -> tuple:
        observation, reward, done, info = self.env.reset()
        self.episode = [observation.copy()]
        return observation, reward, done, info
