import time
from typing import Union
from threading import Event
from multiprocessing import Queue, Lock

import cv2
import numpy as np

from core.control.edge_finder import EdgeFinder, TraditionalEdgeFinder
from core.utils.executions import BaseThreadExec, BaseProcessExec
from core.utils.ipc_utils import put_latest_in_queue
from core.qcar import PhysicalCar, VirtualCSICamera, VirtualRGBDCamera
from .modules import EdgeFinderWrapper, ObserverWrapper
from .factory import PIDControlCarFactory


# TODO: find a way to use other types of EdgeFinder
class EdgeFinderExec(BaseProcessExec):
    def create_instance(self) -> EdgeFinderWrapper:
        edge_finder: EdgeFinder = TraditionalEdgeFinder()
        return EdgeFinderWrapper(edge_finder)


class ObserveExec(BaseProcessExec):
    def create_instance(self) -> ObserverWrapper:
        return ObserverWrapper()


class EdgeFinderComm(BaseThreadExec):
    def setup_thread(self) -> None:
        self.camera: VirtualCSICamera = VirtualCSICamera(id=3)

    def execute(self, edge_response_queue: Queue, observer_response_queue: Queue) -> None:
        image: np.ndarray = self.camera.read_image()
        if image is not None:
            # print("Sending image to edge finder...")
            put_latest_in_queue(image.copy(), edge_response_queue)
            put_latest_in_queue(image.copy(), observer_response_queue)
            # cv2.imshow('EdegFinder Image', image)
            # cv2.waitKey(1)
        else:
            time.sleep(0.001) # thread yielding


class ObserveComm(BaseThreadExec):
    def setup_thread(self) -> None:
        self.camera: VirtualRGBDCamera = VirtualRGBDCamera()

    def execute(self, observe_response_queue: Queue) -> None:
        image: np.ndarray = self.camera.read_rgb_image()
        if image is not None:
            put_latest_in_queue(image.copy(), observe_response_queue)
            # cv2.imshow('Observer Image', image)
            # cv2.waitKey(1)
        else:
            time.sleep(0.001) # thread yielding


class CarComm(BaseThreadExec):
    def setup_thread(self) -> None:
        self.car: PhysicalCar = PIDControlCarFactory().build_car()

    def execute(self, edge_request_queue: Queue, observe_request_queue: Queue) -> None:
        self.car.execute(edge_request_queue, observe_request_queue)

    def run_thread(self, edge_request_queue: Queue, observe_request_queue: Queue) -> None:
        super().run_thread( edge_request_queue, observe_request_queue)
        self.car.halt_car()
