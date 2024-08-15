import os

import numpy as np

from restful.repository import DataRepository


def test_handle_step_complete_1():
    data_repo = DataRepository()
    data_repo.handle_step_complete(0.5, [0.1, 0.2])
    data_repo.episode_data["intervention"] = [1]
    data_repo.episode_data["image"] = [np.zeros((10, 10, 3))]
    path: str = data_repo.handle_episode_complete()
    assert os.path.exists(f"{path}_agent.npz"), "Failed to generate the agent file"
    assert os.path.exists(f"{path}_human.npz"), "Failed to generate the agent file"

    assert data_repo.episode_data == {}, "Failed to reset the episode data"
    assert data_repo.rewards == [], "Failed to reset the rewards"
    assert data_repo.actions == [], "Failed to reset the actions"
    os.remove(f"{path}_agent.npz") # remove the mock agent file
    os.remove(f"{path}_human.npz") # remove the mock human file


def test_handle_step_complete_2():
    data_repo = DataRepository()
    data_repo.handle_step_complete(0.5, [0.1, 0.2])
    data_repo.episode_data["intervention"] = [0]
    data_repo.episode_data["image"] = [np.zeros((10, 10, 3))]
    path: str = data_repo.handle_episode_complete()
    assert os.path.exists(f"{path}_agent.npz"), "Failed to generate the agent file"

    assert data_repo.episode_data == {}, "Failed to reset the episode data"
    assert data_repo.rewards == [], "Failed to reset the rewards"
    assert data_repo.actions == [], "Failed to reset the actions"
    os.remove(f"{path}_agent.npz") # remove the mock agent file