import time
import pickle
from typing import Dict
from threading import Lock
from abc import ABC, abstractmethod

import numpy as np

from arena import env, RealWorldEnv
from ego_state import relay, relay_thread, RelayExecutor
from .repository import IDataRepository, DataRepository


class IDataService(ABC):
    @abstractmethod
    def handle_step_complete(self, data: bytes) -> None:
        ...

    @abstractmethod
    def handle_upload_step_data(self, data: bytes) -> None:
        ...

    @abstractmethod
    def handle_episode_complete(self) -> None:
        ...


class DataService(IDataService):
    def __init__(self) -> None:
        self.lock: Lock = Lock()
        self.env: RealWorldEnv = env
        self.relay: RelayExecutor = relay
        self.repository: IDataRepository = DataRepository() 
        self.__init_env()

    def __init_env(self) -> None:
        relay_thread.start() # start the relay thread
        _, reward, action = self.env.reset(self.relay) # reset the environment
        self.repository.handle_step_complete(reward, action) 

    def handle_step_complete(self, data: bytes) -> None:
        action: np.ndarray = pickle.loads(data)
        done, reward, action = self.env.step(action, self.relay)
        print(f"Action: {action}, Reward: {reward}, Collision: {done}")
        self.repository.handle_step_complete(reward, action)
        if done:
            print("Collision detected!")
            self._handle_reset_env() 

    def _handle_reset_env(self) -> None:
        print("Resetting the environment...")
        self.relay.set_early_stop(True, self.lock) 
        time.sleep(1)         
        self.relay.set_early_stop(False, self.lock)           

    def handle_upload_step_data(self, data: bytes) -> None:
        step_data: Dict[str, np.ndarray] = pickle.loads(data)
        self.repository.handle_upload_step_data(step_data)

    def handle_episode_complete(self) -> None:
        _, reward, action = env.reset(self.relay)  
        self.repository.handle_episode_complete()
        print("Add initial step data to the repository...")
        self.repository.handle_step_complete(reward, action)     
     
