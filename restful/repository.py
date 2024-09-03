import os
import time
import uuid
import logging
from copy import deepcopy
from typing import List, Dict, Any
from abc import ABC, abstractmethod

import numpy as np
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection

from system.settings import NPZ_DIR, HITL_REWARD


class IDataRepository(ABC):
    @abstractmethod
    def handle_step_complete(self, reward: float, action: np.ndarray) -> None:
        ...

    @abstractmethod
    def handle_upload_step_data(self, step_data: Dict[str, np.ndarray]) -> None:
        ...

    @abstractmethod
    def handle_episode_complete(self) -> None:
        ...


class DataRepository(IDataRepository): # DataModel
    def __init__(self):
        self.rewards: List[float] = []
        self.actions: List[np.ndarray] = []
        self.episode_data: Dict[str, List[np.ndarray]] = {}

    def handle_step_complete(self, reward: float, action: np.ndarray) -> None:
        # print(f"Action: {action}, Reward: {reward}")
        self.rewards.append(reward)
        self.actions.append(action)

    def handle_upload_step_data(self, step_data: Dict[str, np.ndarray]) -> None:
        for key, val in step_data.items():
            if key not in self.episode_data.keys():
                self.episode_data[key] = []
            self.episode_data[key].append(val)
        logging.info(f"Added step data: {step_data}")

    def handle_episode_complete(self) -> None:
        episode_uid: str = str(uuid.uuid1(int(time.time() * 1000)))            
        filename: str = os.path.join(NPZ_DIR, f"{episode_uid}")        

        try: 
            print("Starting to pack the episode data...")
            self.episode_data["reward"] = deepcopy(self.rewards)
            self.episode_data["action"] = self.actions
            for key, val in self.episode_data.items():
                if type(val) is not list:
                    continue

                if len(val) < len(self.rewards):
                    self.episode_data[key] = [self.episode_data[key][0]] + self.episode_data[key]

            interventions: np.ndarray = self.episode_data["intervention"]        
            # calculate the agent reward    
            print(len(self.episode_data["reward"]))
            print(len(self.episode_data["image"]))
            print(len(interventions))
            for i in range(len(self.episode_data["reward"])):
                self.episode_data["reward"][i] = self.episode_data["reward"][i] - HITL_REWARD * interventions[i]
                    
            print(f"Episode steps: {len(self.episode_data['reward'])}")
            print("Last step reward: ", self.episode_data["reward"][-1])                
            print(f"Episode reward: {sum(self.episode_data['reward'])}")
            self.episode_data["sentinel"] = True  # for data integrity checking
            with open(f"{filename}_agent.npz", 'wb') as f:
                np.savez(f, **self.episode_data)
            logging.info(f"{len(self.episode_data['image'])} steps written to agent_{filename}")

            if 1 in interventions:
                for i, intervention in enumerate(interventions):
                    self.episode_data["action"][i][0] = self.episode_data["action"][i][0] * (1 - intervention)
                    self.episode_data["reward"][i] = self.rewards[i] + HITL_REWARD * interventions[i]
                with open(f"{filename}_human.npz", 'wb') as f:
                    np.savez(f, **self.episode_data)
                logging.info(f"{len(self.episode_data['image'])} steps written to human_{filename}")
        finally:
            self.episode_data = {}
            self.rewards = []
            self.actions = []

            return f"{filename}"
    

class DatasetRepository:
    def __init__(self) -> None:
        client: MongoClient = MongoClient("mongodb://localhost:27017/")
        self.databse: Database = client["qcardb"]
        self.collection: Collection = self.databse["collisionAviodance"]

    def save_to_npz(self, data: Dict[str, np.ndarray]) -> str:
        episode_uid: str = str(uuid.uuid1(int(time.time() * 1000)))
        filename: str = os.path.join(NPZ_DIR, f"{episode_uid}.npz")
        np.savez_compressed(filename, **data)
        return filename

    def save_to_db(self, filename: str, data: Dict[str, np.ndarray]) -> None:
        if self.collection.find_one({"npz_path": filename}):
            print(f"Data already saved to database")
            return

        intervented_steps: int = [i for i in range(len(data["intervention"])) if data["intervention"][i] == 1]
        final_step_index: int = len(data["reward"]) - 1
        collision_step: int = final_step_index if data["reward"][final_step_index] <= -30 else -999

        data_for_save: Dict[str, Any] = {
            "npz_path": filename,
            "intervention_steps": intervented_steps,
            "collision_step": collision_step,
        }
        self.collection.insert_one(data_for_save)
        print(f"Data saved to database")

    def update_npz_reward(self, filename: str) -> None:
        db_data: str = self.collection.find_one({"npz_path": filename})
        intervention_steps: List[int] = db_data["intervention_steps"]
        collision_step: int = db_data["collision_step"]
        npz_path: str = db_data["npz_path"]
        
        episode_data: Dict[str, np.ndarray] = self.read_from_npz(npz_path)
        npz_type: str = filename[-9:-4]
        for index in intervention_steps:
            episode_data["reward"][index] += -1 if npz_type == "agent" else 1
        if collision_step != -999:
            episode_data["reward"][collision_step] = episode_data["reward"][collision_step] - 20
        self.update_npz_by_filename(npz_path, episode_data)

    def update_npz_by_filename(self, filename: str, data: Dict[str, np.ndarray]) -> None:
        np.savez(filename, **data)

    def read_from_npz(self, filename: str) -> Dict[str, np.ndarray]:
        loaded_data: Dict[str, np.ndarray] = np.load(filename, allow_pickle=True)
        return {key: np.array(value, copy=True) for key, value in loaded_data.items()}

    def read_from_db_by_filename(self, filename: str) -> Dict[str, Any]:
        return self.collection.find_one({"npz_path": filename})
    
    def get_latest_data_list(self) -> None:
        return self.collection.find().sort("_id", -1).limit(347)