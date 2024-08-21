import os
import glob
import pickle   
from typing import List, Dict, Tuple

import cv2
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
    
    def read_npz_files(self, i: int, j: int) -> list:
        available_data: list = []
        for npz_file in self.file_list:
            try: 
                with np.load(npz_file) as data:
                    data['sentinel'] # test the data integrity
                    # data_dict = {file: data[file] for file in data.files}
                    # for image in data_dict['image']:
                    #     cv2.resize(image, (84, 84))
                    #     cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    # available_data.append(data_dict)
            except Exception as e:
                print(f"{npz_file} Corrupted, skipping...")
                continue

        return available_data
    
    def convert_to_traj(self, data: dict) -> dict:
        traj: Dict[str, List[np.ndarray]] = {
            'next_observations': [],
            'observations': [],
        }

        # reward and action
        traj['rewards'] = data['reward']
        traj['actions'] = data['action']  
        # observations (image and state_info)
        for i, image in enumerate(data['image']):
            state: np.ndarray = image.reshape(-1)
            state = np.concatenate((state, data['state_info'][i]))
            state = state.astype(np.float16)
            traj['observations'].append(state)
        traj['observations'] = np.array(traj['observations'])

        # next_observations
        traj['next_observations'] = np.zeros_like(traj['observations'])
        for i in range(len(traj['observations']) - 1):
            traj['next_observations'][i] = traj['observations'][i + 1]

        # calculate the return_to_go
        accumulated_reward: float = 0        
        return_to_go: np.ndarray = np.zeros_like(traj['rewards'])
        for i in range(len(traj['rewards']) - 1, -1, -1):
            return_to_go[i] = accumulated_reward            
            accumulated_reward += traj['rewards'][i]
        traj['return_to_go'] = return_to_go

        return traj
        
    def execute(self) -> None:
        self.read_file_list()
        
        data_list: List[dict] = self.read_npz_files(0, 0)
        trajectories: List[Dict[str, np.ndarray]] = []
        for data in data_list:
            if data is None:
                continue

            traj: Dict[str, np.ndarray] = self.convert_to_traj(data)
            # trajectories.append(traj)
        
        # print(f"Saving trajectories.pkl...")
        # with open(f"assets/model/trajectories.pkl", "wb") as f:
        #     pickle.dump(trajectories, f) 
            
        print(f"trajectories.pkl saved. Releasing memory...")


def run_reinformer_util():
    project_path: str = os.getcwd()
    print(f"Current working directory: {project_path[:-11]}")
    local_path: str = r"assets/npz"
    npz_folder_path: str = os.path.join(project_path[:-11], local_path)
    data_converter: DataConverter = DataConverter(local_path)
    data_converter.execute()

    # with open("assets/trajectories.pkl", "rb") as f:
    #     data = pickle.load(f)
    # print(len(data))


def format_numbers(numbers):
    formatted_numbers = ", ".join(f"{num:.8e}" for num in numbers)
    return formatted_numbers
