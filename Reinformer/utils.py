import os
import glob
import pickle   
from typing import List, Dict

import numpy as np


class DataConverter:
    def __init__(self, folder_path: str) -> None:
        project_path: str = os.getcwd()
        self.counter = 0
        # folder_path is the relative path from the project folder
        self.folder_path: str = os.path.join(project_path, folder_path)
        self.file_list: list = []

    def read_file_list(self) -> None:
        print(f"Reading files in {self.folder_path}")
        search_path: str = self.folder_path + '/*.npz'
        self.file_list = glob.glob(search_path)
        print(f"Found {len(self.file_list)} files")
        self.file_list.sort(key=os.path.getmtime)
    
    def read_npz_files(self) -> list:
        available_data: list = []
        for npz_file in self.file_list:
            try: 
                data = np.load(npz_file, allow_pickle=True)
                available_data.append(data)
                self.counter += 1
            except Exception as e:
                # available_data.append(None)
                print(f"{npz_file} Corrupted, skipping...")
                continue

        return available_data
    
    def convert_to_traj(self, data: dict) -> dict:
        traj: Dict[str, np.ndarray] = {}

        # concatenate the waypoints and tasks
        waypoints: np.ndarray = data['waypoints']
        waypoints = waypoints.reshape(waypoints.shape[0], waypoints.shape[1] * waypoints.shape[2])
        state: np.ndarray = data['state']
        task: np.ndarray = np.array(data['task'])
        place_holder: np.ndarray = np.full((len(data['task']), 15 - len(data['task'][0])), -99)
        # print(f'waypoints shape: {waypoints}, task shape: {task} placeholder shape: {place_holder}')
        state = np.concatenate((waypoints, state, task, place_holder), axis=1)
        # print(f'state shape: {state}')
        # state: np.ndarray = waypoints.reshape(waypoints.shape[0], 1)
        
        # print(f'shape of state: {state.shape}')
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
    print(f"Current working directory: {project_path}")
    local_path: str = r"test_data\npzs"
    npz_folder_path: str = os.path.join(project_path, local_path)
    data_converter: DataConverter = DataConverter(npz_folder_path)
    trajectories: List[Dict[str, np.ndarray]] = data_converter.execute()
    print(f"Number of trajectories: {len(trajectories)}")
    with open("assets/trajectories.pkl", "wb") as f:
        pickle.dump(trajectories, f)
    # with open("assets/trajectories.pkl", "rb") as f:
    #     data = pickle.load(f)
    # print(len(data))

def format_numbers(numbers):
    formatted_numbers = ", ".join(f"{num:.8e}" for num in numbers)
    return formatted_numbers
    