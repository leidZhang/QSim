import time
import pickle
from socket import *
from typing import List, Any
# from multiprocessing import Queue

import numpy as np

from qvl.qlabs import QuanserInteractiveLabs
from core.qcar import QCAR_ACTOR_ID
from core.templates import LifeCycleWrapper
from core.qcar.virtual import Monitor, VirtualRuningGear
from core.remote import UDPServer, UDPClient
from system.settings import DT, IP, PORTS, ENV_PORT


class EgoStateRelay:
    def __init__(self) -> None:
        self.early_stop: bool = False
        self.ego_states: List[np.ndarray] = [np.zeros(6) for _ in range(len(PORTS))]
        self.qlabs: QuanserInteractiveLabs = QuanserInteractiveLabs()
        self.qlabs.open("localhost") # connect to the server
        self.monitors: List[Monitor] = [
            Monitor(QCAR_ACTOR_ID, i, DT) for i in range(len(PORTS))
        ]
        self.servers: List[UDPServer] = [
            UDPServer(IP, PORTS[i]) for i in range(len(PORTS))
        ] # severs to send the states to the actors
        # self.env_server: UDPServer = UDPServer(IP, ENV_PORT) # server to give reward

    def _handle_send_data(self, server: UDPServer, data: Any) -> None:
        serialized_data: bytes = pickle.dumps(data)  
        server.server_socket.sendto(serialized_data, server.address)

    def handle_read_ego_state(self) -> None:
        state: np.ndarray = np.full(6, np.nan)
        if not self.early_stop:
            self.monitors[0].read_state(self.qlabs)
            state = self.monitors[0].state
        self._handle_send_data(self.servers[0], state)
        self.ego_states[0] = state
        # print(f"Car 0: {state}")

    def handle_read_bot_state(self) -> None:
        for i in range(1, len(PORTS)):
            self.monitors[i].read_state(self.qlabs)
            state: np.ndarray = self.monitors[i].state
            self._handle_send_data(self.servers[i], state)
            self.ego_states[i] = state   
            # print(f"Car {i}: {state}")

    def handle_reset_signal(self) -> None:
        self._handle_send_data(self.servers[0], None)

    def get_ego_states(self) -> List[np.ndarray]:
        return self.ego_states

    def terminate(self) -> None:
        for i in range(len(self.servers)):
            self.servers[i].server_socket.close()
        self.qlabs.close()
        