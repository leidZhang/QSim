import logging
from socket import *
from typing import Tuple
from multiprocessing import Queue

import cv2
import numpy as np

from core.utils.ipc_utils import fetch_latest_in_queue


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

    def execute(self, data_queue: Queue, quality: int = 100) -> None:
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
