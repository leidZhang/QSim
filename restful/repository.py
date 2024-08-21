import os
import time
import uuid
import logging
from copy import deepcopy
from typing import List, Dict
from abc import ABC, abstractmethod

import numpy as np

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
    