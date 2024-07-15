import sys
import random
import time
from queue import Queue
from multiprocessing import Process

import numpy as np

from core.remote.server import UDPServer
from core.remote.client import UDPClient


def generate_random_dict() -> np.ndarray:
    data = []
    for i in range(6):
        rand: float = random.uniform(-6, 6)
        data.append(rand)

    return np.array(data)


def push_to_queue(data_queue: Queue) -> np.ndarray:
    data: np.ndarray = generate_random_dict()
    if data_queue.full():
        data_queue.get()
    data_queue.put(data)
    return data


def run_mock_server() -> None:
    data_queue: Queue = Queue(5)
    server: UDPServer = UDPServer(ip='127.0.0.1', port=8080)
    while True:
        data: np.ndarray = push_to_queue(data_queue)
        server.execute(data_queue)
        print(f"Sent data {data}, size: {sys.getsizeof(data)}")


def run_mock_client() -> None:
    data_queue: Queue = Queue(5)
    client: UDPClient = UDPClient(ip='127.0.0.1', port=8080)

    while True:
        client.execute("200", data_queue)
        print(f"Received {data_queue.get()} from the server")


def test_client_server() -> None:
    # try:
        server_process: Process = Process(target=run_mock_server)
        client_process: Process = Process(target=run_mock_client)
        server_process.start()
        client_process.start()

        while True:
            time.sleep(100)
    # except KeyboardInterrupt:
    #     print("Stop test")
    # except Exception as e:
    #     print(e)
