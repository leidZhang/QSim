import time
import pickle
from typing import List
from queue import Queue

import numpy as np

from qvl.qlabs import QuanserInteractiveLabs
from core.qcar import QCAR_ACTOR_ID
from core.qcar.virtual import Monitor, VirtualRuningGear
from core.remote import UDPServer, UDPClient
from system.settings import DT, IP, PORTS


def run_car() -> None:
    qlabs: QuanserInteractiveLabs = QuanserInteractiveLabs()
    qlabs.open("localhost") # connect to the server
    running_gears: List[VirtualRuningGear] = [
        VirtualRuningGear(QCAR_ACTOR_ID, i) for i in range(len(PORTS))
    ]
    while True:
        for i in range(len(running_gears)):
            running_gears[i].read_write_std(qlabs, 0.08, 0.0)


def run_comm() -> None:
    qlabs: QuanserInteractiveLabs = QuanserInteractiveLabs()
    qlabs.open("localhost") # connect to the server
    monitors: List[Monitor] = [
        Monitor(QCAR_ACTOR_ID, i, DT) for i in range(len(PORTS))
    ]
    servers: List[UDPServer] = [
        UDPServer(IP, PORTS[i]) for i in range(len(PORTS))
    ]    

    while True:
        start: float = time.time()
        for i in range(len(PORTS)):
            monitors[i].read_state(qlabs)
            state: np.ndarray = monitors[i].state
            serialized_data: bytes = pickle.dumps(state)
            servers[i].server_socket.sendto(serialized_data, servers[i].address)
            # print(f"Car {i} state: {monitors[i].state}")
        end: float = time.time() - start
        time.sleep(max(DT - end, 0))


def run_client() -> None:
    state_queue: Queue = Queue(1)
    client: UDPClient = UDPClient(IP, PORTS[0])
    while True:
        client.execute(state_queue)
        print(f"Client state: {state_queue.get()}")
        