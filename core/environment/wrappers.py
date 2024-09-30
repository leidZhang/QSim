from typing import Tuple

import numpy as np
from gym import Wrapper

from .environment import QLabEnvironment


class ActionRewardResetWrapper(Wrapper):
    """
    ActionRewardResetWrapper class is A wrapper class for the Env classes, responsible for
    adding more data in the observation dict.
    """

    def step(
        self,
        action: np.ndarray,
        metrics: np.ndarray,
        compare_action: np.ndarray
    ) -> Tuple[dict, float, bool, dict]:
        """
        Steps the environment with the given action.

        Parameters:
        - action (np.ndarray): The action to take
        - metrics (np.ndarray): The metrics
        - compare_action (np.ndarray): The compare action

        Returns:
        - tuple: The observation, reward, done, and info
        """
        observation, reward, done, info = self.env.step(action=action, metrics=metrics, compare_action=compare_action)
        observation["action"] = np.array(action, dtype=np.float32)
        observation["reward"] = np.array(reward, dtype=np.float32)
        observation["terminal"] = np.array(done, dtype=bool)
        observation["reset"] = np.array(False, dtype=bool)

        return observation, reward, done, info

    def reset(self) -> Tuple[dict, float, bool, dict]:
        """
        Resets the environment and add additional information to the observation.

        Returns:
        - tuple: The observation, reward, done, and info
        """
        observation, reward, done, info = self.env.reset()
        observation["action"] = np.zeros((self.action_size, ), dtype=np.float32)
        observation["reward"] = np.array(0.0, dtype=np.float32)
        observation["terminal"] = np.array(False, dtype=bool)
        observation["reset"] = np.array(True, dtype=bool)

        return observation, reward, done, info


class CollectionWrapper(Wrapper):
    """
    CollectionWrapper class is a wrapper class for the Env classes, responsible for
    reorginizing the observation dict.

    Attributes:
    - env (QLabEnvironment): The QLabEnvironment object
    - paddings (dict): The paddings for the observation
    - episode (list): The list of episodes
    """

    def __init__(self, env: QLabEnvironment, paddings: dict = {}) -> None:
        """
        Initializes the CollectionWrapper object.

        Parameters:
        - env (QLabEnvironment): The QLabEnvironment object
        - paddings (dict): The paddings for the observation
        """
        super().__init__(env)
        self.paddings: dict = paddings
        self.episode: list = []

    def step(
        self,
        action: np.ndarray,
        metrics: np.ndarray,
        compare_action: np.ndarray
    ) -> Tuple[dict, float, bool, dict]:
        """
        Steps the environment with the given action.

        Parameters:
        - action (np.ndarray): The action to take
        - metrics (np.ndarray): The metrics
        - compare_action (np.ndarray): The compare action

        Returns:
        - tuple: The observation, reward, done, and info
        """
        observation, reward, done, info = self.env.step(
            action=action, metrics=metrics, compare_action=compare_action
        )
        # copy obs dict as a item and add it to self.episode list
        self.episode.append(observation.copy())
        if not done:
            return observation, reward, done, info

        episode: dict = {}
        for k in self.episode[0]:
            data: list = []
            for t in self.episode:
                if k in self.paddings:
                    shape, value = self.paddings[k]
                    data.append(np.pad(
                        t[k], ((0, shape[0] - t[k].shape[0]), (0, 0)),
                        mode='constant', constant_values=(value, )
                    ))
                else:
                    data.append(t[k])
            episode[k] = np.array(data)

        info["episode"] = episode
        return observation, reward, done, info

    def reset(self) -> Tuple[dict, float, bool, dict]:
        """
        Resets the environment and add additional information to the observation.

        Returns:
        - tuple: The observation, reward, done, and info
        """
        observation, reward, done, info = self.env.reset()
        self.episode = [observation.copy()]
        return observation, reward, done, info
