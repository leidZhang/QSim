import sys
import time
import pickle
import logging
from socket import *
from typing import Tuple, Any, Union
from queue import Queue
from multiprocessing import Queue as MPQueue

import cv2
import numpy as np

from core.utils.ipc_utils import fetch_latest_in_queue


class UDPServer:
    def __init__(self, ip: str = '0.0.0.0', port: int = 8080) -> None:
        self.server_socket: socket = socket(AF_INET, SOCK_DGRAM)
        self.address: Tuple[str, int] = (ip, port)

    def _send_data(self, data_queue: Union[Queue, MPQueue]) -> None:
        if data_queue.empty():
            time.sleep(0.001)
            return

        data: Any = data_queue.get()
        serialized_data: bytes = pickle.dumps(data)
        self.server_socket.sendto(serialized_data, self.address)

    def _receive_data(self) -> Any:
        serialized_data, _ = self.server_socket.recvfrom(1024)
        return pickle.loads(serialized_data)
    
    def execute(self, data_queue: Union[Queue, MPQueue]) -> None:
        # try:
            self._send_data(data_queue)
            # self._receive_data()
        # except Exception as e:
        #     logging.warning(str(e))


# TODO: Change to UDP for faster transmission
class VideoServer:
    def __init__(self, port: int = 8080) -> None:
        self.server_socket: socket = socket(AF_INET, SOCK_STREAM)
        self.address: Tuple[str, int] = ('0.0.0.0', port)

    def setup(self) -> None:
        self.server_socket.bind(self.address)

    def send_image(self, image: np.ndarray, quality: int) -> None:
        if image is None:
            return

        # encode the input image
        _, send_data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        # send image to the client
        self.server_socket.sendall(send_data.tobytes())

    def execute(self, data_queue: MPQueue, quality: int = 100) -> None:
        # wait for client to connect to the server
        logging.info("The video server module is ready to accept connection...")
        self.client_socket, self.client_address = self.server_socket.accept()
        logging.info(f"Connected to {self.client_address}")

        # transmit image to the client
        client_is_alive: bool = True
        while client_is_alive:
            try:
                # fetch image from the multiprocessing queue
                image: np.ndarray = fetch_latest_in_queue(data_queue, quality)
                # send image to the client
                self.send_image(image=image)
            except Exception as e:
                logging.warning(str(e))
                client_is_alive = False

        # close the connection if there's any issues
        self.client_address = ''
        self.client_socket.close()
