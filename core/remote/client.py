import sys
import pickle
import logging
from socket import *
from typing import Tuple, Union, Any
from queue import Queue
from multiprocessing import Queue as MPQueue


class UDPClient:
    def __init__(self, ip: str, port: int = 8080) -> None:
        print("Preparing receiving data...")
        self.client_socket: socket = socket(AF_INET, SOCK_DGRAM)
        self.address: Tuple[str, int] = (ip, port)
        self.client_socket.bind((ip, port))


    def _receive_data(self, data_queue: Union[Queue, MPQueue]) -> None:
        serialized_data, _ = self.client_socket.recvfrom(1024)
        data = pickle.loads(serialized_data)
        data_queue.put(data)

    def _send_data(self, data: Any) -> None:
        serialized_data: bytes = pickle.dumps(data)
        self.client_socket.sendto(serialized_data, self.address)

    def execute(
        self, 
        response_data: Any, 
        data_queue: Union[Queue, MPQueue]
    ) -> None:
        # try:
            self._receive_data(data_queue)
            # self._send_data(response_data)
        # except Exception as e:
        #     logging.warning(str(e))
