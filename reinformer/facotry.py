import pickle
from typing import Any, Dict, List

import numpy as np

from .dataset import SimplifiedD4RLDataset, discount_cumsum


# class SimplifiedD4RLDatasetFactory:
#     def __init__(self) -> None:
#         self.scales: Dict[str, int] = {
#             "hopper": 1000, "walker2d": 1000,
#             "halfcheetah": 5000, "maze2d": 100,
#             "kitchen": 100, "pen": 10000,
#             "door": 10000, "hammer": 10000,
#             "relocate": 10000, "antmaze": 1
#         } # add more scales as needed
    
#     def _load_dataset(self, dataset_path: str) -> dict:
#         with open(dataset_path, "rb") as f:
#             trajectories: dict = pickle.load(f)
#         return trajectories
    
#     def _determine_scale(self, env_name: str) -> int:
#         if env_name not in self.scales:
#             raise ValueError(f"Environment {env_name} not found in scales")
#         return self.scales[env_name]
    
#     def _process_trajectories(
#             self, 
#             dataset: SimplifiedD4RLDataset, 
#             dataset_path: str, 
#             scale: int
#         ) -> list:
#         states, returns = [], []
#         for traj in dataset.trajectories:
#             if "antmaze" in dataset_path:
#                 traj["rewards"] = traj["rewards"] * 100 + 1
#             states.append(traj["observations"])
#             returns.append(traj["rewards"].sum())
#             traj["returns_to_go"] = (
#                 discount_cumsum(traj["rewards"], 1) / scale
#             )
#         states = np.concatenate(states, axis=0)
#         dataset.state_mean, dataset.state_std = (
#             np.mean(states, axis=0),
#             np.std(states, axis=0) + 1e-6,
#         )
#         return returns

#     def _normalize_states(self, dataset: SimplifiedD4RLDataset) -> None:
#         for traj in dataset.trajectories:
#             traj["observations"] = (
#                 traj["observations"] - dataset.state_mean
#             ) / dataset.state_std

#             traj["next_observations"] = (
#                 traj["next_observations"] - dataset.state_mean
#             ) / dataset.state_std

#     def _calculate_return_stats(self, dataset: SimplifiedD4RLDataset, return_list: list) -> None:
#         returns: np.ndarray = np.array(return_list)
#         max_return, mean_return, std_return = returns.max(), returns.mean(), returns.std()
#         dataset.return_stats = [
#             max_return,
#             mean_return,
#             std_return
#         ]
#         print(f"dataset size: {len(dataset.trajectories)}\nreturns max :\
#             {max_return}\nreturns mean: {mean_return}\nreturns std :\
#             {std_return}"
#         )

#     def create_dataset(
#         self,
#         env_name: str, 
#         dataset_path: str, 
#         context_len: int, 
#         device: str
#     ) -> SimplifiedD4RLDataset:
#         # create the D4RLTrajectoryDataset object
#         dataset: SimplifiedD4RLDataset = SimplifiedD4RLDataset(context_len, device)
#         # load the dataset trajectories
#         dataset.trajectories = self._load_dataset(dataset_path)
#         # determine the reward scale by the environment name
#         scale: int = self._determine_scale(env_name)
#         # process the trajectories and return the list of returns
#         return_list: list = self._process_trajectories(dataset, dataset_path, scale)
#         # normalize the states
#         self._normalize_states(dataset)
#         # calculate the max, mean, and std of the returns and set the dataset.return_stats
#         self._calculate_return_stats(dataset, return_list)
#         return dataset
    