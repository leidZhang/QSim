import os
import glob
import pickle   
from typing import List, Dict

import numpy as np


class DataConverter:
    def __init__(self, folder_path: str) -> None:
        project_path: str = os.getcwd()
        # folder_path is the relative path from the project folder
        self.folder_path: str = os.path.join(project_path, folder_path)
        self.file_list: list = []

    def read_file_list(self) -> None:
        print(f"Reading files in {self.folder_path}")
        search_path: str = self.folder_path + '/*.npz'
        self.file_list = glob.glob(search_path)
        self.file_list.sort(key=os.path.getmtime)
    
    def read_npz_files(self) -> list:
        available_data: list = []
        for npz_file in self.file_list:
            try: 
                data = np.load(npz_file)
                for file in data.files:
                    data[file] # test the data integrity
                available_data.append(data)
            except Exception as e:
                available_data.append(None)
                print(f"{npz_file} Corrupted, skipping...")
                continue

        return available_data
    
    def convert_to_traj(self, data: dict) -> dict:
        traj: Dict[str, np.ndarray] = {}

        # calculate the next_state
        state: np.ndarray = data['state']
        next_state: np.ndarray = np.zeros_like(state)
        for i in range(len(state) - 1):
            next_state[i] = state[i + 1]
        # calculate the return_to_go
        reward: np.ndarray = data['reward']
        return_to_go: np.ndarray = np.zeros_like(reward)
        accumulated_reward: float = 0
        for i in range(len(reward) - 1, -1, -1):
            accumulated_reward += reward[i]
            return_to_go[i] = accumulated_reward
        # assign the data to the dictionary
        traj['observations'] = state
        traj['next_observations'] = next_state
        traj['rewards'] = reward
        traj['returns_to_go'] = return_to_go
        traj['actions'] = data['action']

        return traj
        
    def execute(self) -> None:
        self.read_file_list()
        data_list: List[dict] = self.read_npz_files()
        trajectories: List[Dict[str, np.ndarray]] = []
        for data in data_list:
            if data is None:
                continue
            traj: Dict[str, np.ndarray] = self.convert_to_traj(data)
            trajectories.append(traj)
        return trajectories

def test_reinformer_util():
    project_path: str = os.getcwd()
    print(f"Current working directory: {project_path[:-11]}")  # 打印当前工作目录
    local_path: str = r"mlruns\0\64f95931665541f0910cd7b58a6a9e56\artifacts\episodes_train\0"
    npz_folder_path: str = os.path.join(project_path[:-11], local_path)
    data_converter: DataConverter = DataConverter(local_path)
    trajectories: List[Dict[str, np.ndarray]] = data_converter.execute()

    with open("assets/trajectories.pkl", "wb") as f:
        pickle.dump(trajectories, f)
    # with open("assets/trajectories.pkl", "rb") as f:
    #     data = pickle.load(f)
    # print(len(data))
    