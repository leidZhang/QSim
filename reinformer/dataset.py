import cv2
import json
import pickle
import random
from typing import Tuple, Dict, List

import torch
import numpy as np

from torch.utils.data import Dataset

from ..settings import *

# SCALES: Dict[str, int] = {  
#     "hopper": 1000, "walker2d": 1000,
#     "halfcheetah": 5000, "maze2d": 100,
#     "kitchen": 100, "pen": 10000,
#     "door": 10000, "hammer": 10000,
#     "relocate": 10000, "antmaze": 1
# } # add more scales as needed


def discount_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    disc_cumsum: np.ndarray = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t + 1]
    return disc_cumsum


class D4RLTrajectoryDataset(Dataset): 
    def __init__(
        self, 
        # env_name: str,
        dataset_path: str, 
        context_len: int, 
        device: str,
    ) -> None:
        self.context_len: int = context_len
        self.device: str = device
        # load dataset
        with open(dataset_path, "rb") as f:
            self.trajectories: list = pickle.load(f)

        # calculate state mean and variance and returns_to_go for all traj
        self._cal_state_mean_and_variance()
        # normalize states
        self._normalize_states()        

    def _cal_state_mean_and_variance(self) -> None:
        states, returns, returns_to_go = [], [], []
        for traj in self.trajectories:
            states.append(traj["observations"])
            returns.append(traj["rewards"].sum())
            # calculate returns to go 
            traj["returns_to_go"] = discount_cumsum(traj["rewards"], 1)
        states: np.ndarray = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = (
            np.mean(states, axis=0),
            np.std(states, axis=0) + 1e-6,
        )
                # calculate returns max, mean, std
        returns: np.ndarray = np.array(returns)
        self.return_stats: list = [
            returns.max(),
            returns.mean(),
            returns.std()
        ]
        print(f"dataset size: {len(self.trajectories)}\nreturns max : {returns.max()}\nreturns mean: {returns.mean()}\nreturns std : {returns.std()}")  

    def _normalize_states(self) -> None:
        for traj in self.trajectories:
            traj["observations"] = (
                traj["observations"] - self.state_mean
            ) / self.state_std

            traj["next_observations"] = (
                traj["next_observations"] - self.state_mean
            ) / self.state_std

    def get_state_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.state_mean, self.state_std
    
    def get_return_stats(self) -> list:
        return self.return_stats

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx) -> tuple: # tuple with 7 tensors
        traj: Dict[int, np.ndarray] = self.trajectories[idx]
        traj_len: int = traj["observations"].shape[0]
        return self._get_data(traj_len, traj)

    def _get_data(self, traj_len: int, traj: dict) -> tuple:
        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si: int = random.randint(0, traj_len - self.context_len)

            states: torch.Tensor = torch.from_numpy(
                traj["observations"][si : si + self.context_len]
            )
            images: torch.Tensor = torch.from_numpy(
                traj["images"][si : si + self.context_len]
            )
            next_states: torch.Tensor = torch.from_numpy(
                traj["next_observations"][si : si + self.context_len]
            )
            actions: torch.Tensor = torch.from_numpy(
                traj["actions"][si : si + self.context_len]
            )
            returns_to_go: torch.Tensor = torch.from_numpy(
                traj["returns_to_go"][si : si + self.context_len]
            )
            rewards: torch.Tensor = torch.from_numpy(
                traj["rewards"][si : si + self.context_len]
            )
            timesteps: torch.Tensor = torch.arange(
                start=si, end=si + self.context_len, step=1
            )

            # all ones since no padding
            traj_mask: torch.Tensor = torch.ones(self.context_len, dtype=torch.long)

        else:
            padding_len: int = self.context_len - traj_len

            # padding with zeros
            states: torch.Tensor = torch.from_numpy(traj["observations"])
            states = torch.cat(
                [
                    states,
                    torch.zeros(
                        ([padding_len] + list(states.shape[1:])),
                        dtype=states.dtype,
                    ),
                ],
                dim=0,
            )

            images: torch.Tensor = torch.from_numpy(traj["images"])
            images = torch.cat(
                [
                    images,
                    torch.zeros(
                        ([padding_len] + images.shape[1:]),
                        dtype=states.dtype,
                    ),
                ],
                dim=0,
            )

            next_states: torch.Tensor = torch.from_numpy(traj["next_observations"])
            next_states = torch.cat(
                [
                    next_states,
                    torch.zeros(
                        ([padding_len] + list(next_states.shape[1:])),
                        dtype=next_states.dtype,
                    ),
                ],
                dim=0,
            )

            actions: torch.Tensor = torch.from_numpy(traj["actions"])
            actions = torch.cat(
                [
                    actions,
                    torch.zeros(
                        ([padding_len] + list(actions.shape[1:])),
                        dtype=actions.dtype,
                    ),
                ],
                dim=0,
            )

            returns_to_go: torch.Tensor = torch.from_numpy(traj["returns_to_go"])
            returns_to_go = torch.cat(
                [
                    returns_to_go,
                    torch.zeros(
                        ([padding_len] + list(returns_to_go.shape[1:])),
                        dtype=returns_to_go.dtype,
                    ),
                ],
                dim=0,
            )

            rewards: torch.Tensor = torch.from_numpy(traj["rewards"])
            rewards = torch.cat(
                [
                    rewards,
                    torch.zeros(
                        ([padding_len] + list(rewards.shape[1:])),
                        dtype=rewards.dtype,
                    ),
                ],
                dim=0,
            )

            timesteps: torch.Tensor = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask: torch.Tensor = torch.cat(
                [
                    torch.ones(traj_len, dtype=torch.long),
                    torch.zeros(padding_len, dtype=torch.long),
                ],
                dim=0,
            )

        return (
            timesteps,
            states,
            images,
            next_states,
            actions,
            returns_to_go,
            rewards,
            traj_mask,
        )

import gc
import os
import glob

import cv2


class CustomDataSet(D4RLTrajectoryDataset):
    def __init__(
        self, 
        dataset_path: str, 
        context_len: int, 
        device: str,
        resume: bool = False
    ) -> None:
        self.resume: bool = resume
        self.context_len: int = context_len
        self.device: str = device
        search_path: str = dataset_path + '/*.npz'
        self.file_list = glob.glob(search_path)
        self.file_list.sort(key=os.path.getmtime)
        self._cal_state_mean_and_variance()

    def _read_file_by_index(self, index: int) -> None:
        npz_file: str = self.file_list[index]
        episode_data: dict = None
        
        try: 
            data = np.load(npz_file, allow_pickle=True)
            episode_data = {file: data[file] for file in data.files}
        except Exception as e:
            print(f"{npz_file} Corrupted, skipping...")

        return episode_data
    
    def _online_mean_and_std(self, observations: np.ndarray, m2: np.ndarray, count: int) -> Tuple[np.ndarray, np.ndarray]:
        for i, observation in enumerate(observations):
            delta: np.ndarray = observation - self.state_mean
            count += 1 # update the counter
            self.state_mean += delta / count
            m2 += delta * (observation - self.state_mean)
        return self.state_mean, m2, count
    
    def _cal_state_mean_and_variance(self) -> None:
        self.state_mean, m2, self.state_std = np.zeros(STATE_DIM), np.zeros(STATE_DIM), np.zeros(STATE_DIM)
        states, returns, returns_to_go = [], [], []

        if not self.resume:
            count: int = 0            
            for i in range(len(self.file_list)):
                episode_data: dict = self._read_file_by_index(i)
                episode_data = self._preprocess_data(episode_data)
            
                self.state_mean, m2, count = self._online_mean_and_std(episode_data["observations"], m2, count)
                # states.append(episode_data["observations"])
                return_to_go = discount_cumsum(episode_data["rewards"], 1)
                returns.append(episode_data["rewards"].sum()) 
                returns_to_go.append(return_to_go)            

            variance: np.ndarray = m2 / count
            self.state_std = np.sqrt(variance) + 1e-6
            returns: np.ndarray = np.array(returns)
            self.return_stats: list = [
                returns.max() if len(returns) > 0 else 0,
                returns.mean() if len(returns) > 0 else 0,
                returns.std() if len(returns) > 0 else 0
            ]            
        else:
            with open("state_stat.json", "r") as f:
                state_stat: dict = json.load(f)
                self.state_mean = np.array(state_stat["state_mean"])
                self.state_std = np.array(state_stat["state_std"])
                self.return_stats = state_stat["return_stats"]

        print(f"dataset size: {len(self.file_list)}\nreturns max : {self.return_stats[0]}\nreturns mean: {self.return_stats[1]}\nreturns std : {self.return_stats[2]}")
    
    # TODO: May have to change this to callback function and move it outside the class
    def _preprocess_data(self, data: dict) -> dict:
        traj: Dict[str, List[np.ndarray]] = {
            'next_observations': [],
            'observations': [],
        }

        # reward and action
        traj['rewards'] = data['reward']
        traj['actions'] = data['action'] 
        # Process images
        processed_images = []
        for image in data["image"]:
            # Split the image into four 84 * 84 * 3 blocks
            blocks = [image[:, i*84:(i+1)*84, :] for i in range(4)]
            # Concatenate the blocks along the last dimension
            processed_image = np.concatenate(blocks, axis=-1)
            processed_images.append(processed_image.transpose(2, 0, 1) / 255 - 0.5)
        traj["images"] = np.array(processed_images)
        # traj["images"] = np.array(traj["images"])
        traj['observations'] = data["state_info"]
        traj['next_observations'] = np.zeros_like(traj['observations'])
        traj['next_observations'][:-1] = traj['observations'][1:]

        # calculate the return_to_go
        accumulated_reward: float = 0        
        return_to_go: np.ndarray = np.zeros_like(traj['rewards'])
        for i in range(len(traj['rewards']) - 1, -1, -1):
            return_to_go[i] = accumulated_reward            
            accumulated_reward += traj['rewards'][i]
        traj['returns_to_go'] = return_to_go

        return traj    
    
    def _normalize_traj_states(self, traj: List[np.ndarray]) -> None:
        traj["observations"] = (
            traj["observations"] - self.state_mean
        ) / self.state_std

        traj["next_observations"] = (
            traj["next_observations"] - self.state_mean
        ) / self.state_std
        
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx) -> tuple:
        traj: Dict[int, np.ndarray] = self._read_file_by_index(idx)
        traj = self._preprocess_data(traj)
        self._normalize_traj_states(traj)
        traj_len: int = traj["observations"].shape[0]
        return self._get_data(traj_len, traj)
    