import pickle
from typing import Dict
from abc import ABC, abstractmethod

import numpy as np

from env import env, RealWorldEnv
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
        self.repository.handle_step_complete(reward, action)
        self.handle_episode_complete() if done else lambda: None

    def handle_upload_step_data(self, data: bytes) -> None:
        step_data: Dict[str, np.ndarray] = pickle.loads(data)
        self.repository.handle_upload_step_data(step_data)

    def handle_episode_complete(self) -> None:
        self.repository.handle_episode_complete()
        _, reward, action = env.reset(self.relay) 
        self.repository.handle_step_complete(reward, action) 
