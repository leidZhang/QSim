import time
import pickle
import logging
from typing import Callable
from socket import *
from typing import Tuple, Any, Union
from queue import Queue
from multiprocessing import Queue as MPQueue

import cv2
import numpy as np

from core.utils.ipc_utils import fetch_latest_in_queue
from ..templates.life_cycle import LifeCycleWrapper


class UDPServer(LifeCycleWrapper):
    """
    UDP Server is a primitive server that sends and receives data using UDP protocol.

    Attributes:
    - server_socket (socket): The server socket object
    - address (Tuple[str, int]): The address of the server
    """

    def __init__(self, ip: str = '0.0.0.0', port: int = 8080) -> None:
        """
        Initializes the UDPServer object. The server socket is created and binded to
        the given address. The default port is 8080.

        Parameters:
        - ip (str): The IP address of the server
        - port (int): The port number of the server
        """
        self.server_socket: socket = socket(AF_INET, SOCK_DGRAM)
        self.address: Tuple[str, int] = (ip, port)

    def _send_data(self, data: Any) -> None:
        """
        Sends data to the client.

        Parameters:
        - data_queue (Union[Queue, MPQueue]): The data queue to get the data from

        Returns:
        - None
        """
        serialized_data: bytes = pickle.dumps(data)
        self.server_socket.sendto(serialized_data, self.address)

    def _receive_data(self) -> Any:
        """
        Receives data from the client.

        Returns:
        - Any: The received data
        """
        serialized_data, _ = self.server_socket.recvfrom(1024)
        return pickle.loads(serialized_data)

    def execute(self, data_queue: Union[Queue, MPQueue]) -> None:
        """
        Executes the server module, sending data to the client.

        Parameters:
        - data_queue (Union[Queue, MPQueue]): The data queue to get the data from

        Returns:
        - None
        """
        try:
            data: Any = data_queue.get()
            self._send_data(data)
            # self._receive_data()
        except Exception as e:
            logging.warning(str(e))


class TCPServer(LifeCycleWrapper):
    def __init__(self, port: int = 8080) -> None:
        self.server_socket: socket = socket(AF_INET, SOCK_STREAM)
        self.address: Tuple[str, int] = ('0.0.0.0', port)

    def setup(self) -> None:
        self.server_socket.bind(self.address)

    def execute(self, data_queue: Union[MPQueue, Queue], response: Callable = lambda *args: "received") -> None:
        # wait for client to connect to the server
        logging.info("The video server module is ready to accept connection...")
        self.client_socket, self.client_address = self.server_socket.accept()
        logging.info(f"Connected to {self.client_address}")

        # transmit image to the client
        client_is_alive: bool = True
        while client_is_alive:
            try:
                # receive the data from the client
                serialized_data: bytes = self.client_socket.recv(1024)
                data: Any = pickle.loads(serialized_data)
                data_queue.put(data) 

                # send the response back to the client
                response_data: Any = response(data)
                serialized_data: bytes = pickle.dumps(response_data)
                self.client_socket.sendall(serialized_data)
            except Exception as e:
                logging.warning(str(e))
                client_is_alive = False

        # close the connection if there's any issues
        self.client_address = ''
        self.client_socket.close()
