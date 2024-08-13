import time
import pickle
from typing import Dict
from queue import Queue
# from threading import Thread, Event
from abc import ABC, abstractmethod
from multiprocessing import Event, Queue, Process

import numpy as np

from arena import RealWorldEnv, setup_env
from ego_state import run_relay
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
        self.collision_event = Event()
        self.env: RealWorldEnv = setup_env(self.collision_event)
        # self.relay: RelayExecutor = relay
        self.state_queue: Queue = Queue(1)
        self.repository: IDataRepository = DataRepository() 
        self.__init_env()

    def __init_env(self) -> None:
        print("Starting relay...")
        relay_process: Process = Process(
            target=run_relay, args=(self.state_queue, self.collision_event)
        )
        relay_process.start() # start the relay thread
        print("Initializing environment...")
        _, _, _ = self.env.reset(self.state_queue) # reset the environment
        # print("Add initial step data to the repository...")
        # self.repository.handle_step_complete(reward, action) 

    def handle_step_complete(self, data: bytes) -> None:
        action: np.ndarray = pickle.loads(data)
        done, reward, action = self.env.step(action, self.state_queue)
        self.repository.handle_step_complete(reward, action)
        if done:
            print("Collision detected!")
            time.sleep(1)
            self.env.reset_ego_vehicle([10, 10, 0], [0, 0, 0])          

    def handle_upload_step_data(self, data: bytes) -> None:
        step_data: Dict[str, np.ndarray] = pickle.loads(data)
        self.repository.handle_upload_step_data(step_data)

    def handle_episode_complete(self) -> None:
        if self.collision_event.is_set():
            self.collision_event.clear()        

        _, reward, action = self.env.reset(self.state_queue)  
        self.repository.handle_episode_complete()
        print("Add initial step data to the repository...")
        self.repository.handle_step_complete(reward, action)    

     
