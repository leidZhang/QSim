import sys
import pickle
import logging
from socket import *
from typing import Tuple, Union, Any
from queue import Queue
from multiprocessing import Queue as MPQueue

from ..templates.life_cycle import LifeCycleWrapper


class UDPClient(LifeCycleWrapper):
    """
    UDP client is a simple client that sends and receives data using UDP protocol.

    Attributes:
    - client_socket (socket): The client socket object
    - address (Tuple[str, int]): The address of the server
    """

    def __init__(self, ip: str, port: int = 8080) -> None:
        """
        Initializes the UDPClient object. The client socket is created and binded to
        the given address. The default port is 8080.

        Parameters:
        - ip (str): The IP address of the server
        - port (int): The port number of the server
        """
        print("Preparing receiving data...")
        self.client_socket: socket = socket(AF_INET, SOCK_DGRAM)
        self.address: Tuple[str, int] = (ip, port)
        self.client_socket.bind((ip, port))

    def _receive_data(self, data_queue: Union[Queue, MPQueue]) -> None:
        """
        Receives data from the server and puts it into the data queue.

        Parameters:
        - data_queue (Union[Queue, MPQueue]): The data queue to put the received data

        Returns:
        - None
        """
        serialized_data, _ = self.client_socket.recvfrom(1024)
        data = pickle.loads(serialized_data)
        data_queue.put(data)

    def _send_data(self, data: Any) -> None:
        """
        Sends data to the server.

        Parameters:
        - data (Any): The data to be sent

        Returns:
        - None
        """
        serialized_data: bytes = pickle.dumps(data)
        self.client_socket.sendto(serialized_data, self.address)

    def execute(self, data_queue: Union[Queue, MPQueue]) -> None:
        """
        Executes the client by receiving data from the server and putting it
        into the data queue.
        """
        try:
            self._receive_data(data_queue)
            # self._send_data(response_data)
        except Exception as e:
            logging.warning(str(e))


class TCPClient(LifeCycleWrapper):
    def __init__(self, ip: str, port: int = 8080) -> None:
        self.client_socket: socket = socket(AF_INET, SOCK_STREAM)
        self.address: Tuple[str, int] = (ip, port)

    def connect(self) -> None:
        self.client_socket.connect(self.address)

    def _send_request(self, data: Any) -> Any:
        serialized_data: bytes = pickle.dumps(data)
        self.client_socket.sendall(serialized_data)
        response_data: bytes = self.client_socket.recv(1024)
        return pickle.loads(response_data)

    def execute(self, data_queue: Union[Queue, MPQueue]) -> Any:
        response_data: Any = None

        try:
            data = data_queue.get()
            response_data = self._send_request(data)
        except Exception as e:
            logging.warning(str(e))
        
        return response_data
