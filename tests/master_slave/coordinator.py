import time
from typing import List, Dict, Tuple
from threading import Thread

import cv2
import numpy as np

from core.qcar import PhysicalCar
from core.qcar import VirtualCSICamera, VirtualRGBDCamera
from .vehicle import PIDControlCar
from .constants import DEFAULT_INTERCEPT_OFFSET, DEFAULT_SLOPE_OFFSET
from .constants import THROTTLE_DEFAULT_K_P, THROTTLE_DEFAULT_K_I, THROTTLE_DEFAULT_K_D
from .constants import STEERING_DEFAULT_K_P, STEERING_DEFAULT_K_I, STEERING_DEFAULT_K_D


# TODO: Refactor this class
class QCarCoordinator:
    def __init__(self, duration: float) -> None:
        # CREATE QUANSER HARDWARE OBJECTS IN A THREAD OR A PROCESS
        self.duration: float = duration

    def start_csi_comm(self) -> None:
        csi_camera: VirtualCSICamera = VirtualCSICamera(id=3)
        start: float = time.time()
        while time.time() - start < self.duration:
            image: np.ndarray = csi_camera.read_image()
            if image is not None:
                cv2.imshow('CSI Image', image)
                cv2.waitKey(1)
            else:
                time.sleep(0.001)

    def start_rgbd_comm(self) -> None:
        rgbd_camera: VirtualRGBDCamera = VirtualRGBDCamera()
        start: float = time.time()
        while time.time() - start < self.duration:
            image: np.ndarray = rgbd_camera.read_rgb_image()
            if image is not None:
                cv2.imshow('RGB Image', image)
                cv2.waitKey(1)
            else:
                time.sleep(0.001)

    # TODO: Change some parameters
    def start_car_comm(self) -> None:
        car: PhysicalCar = PIDControlCar(throttle_coeff=1, steering_coeff=1)
        offsets: Tuple[float, float] = (DEFAULT_SLOPE_OFFSET, DEFAULT_INTERCEPT_OFFSET)
        pid_gains: Dict[str, List[float]] = {
            'steering': [STEERING_DEFAULT_K_P, STEERING_DEFAULT_K_I, STEERING_DEFAULT_K_D],
            'throttle': [THROTTLE_DEFAULT_K_P, THROTTLE_DEFAULT_K_I, THROTTLE_DEFAULT_K_D]
        }
        car.setup(pid_gains=pid_gains, offsets=offsets)

        start: float = time.time()
        last_reset: float = start
        while time.time() - start < self.duration:
            if time.time() - last_reset > 2:
                last_reset = time.time()
            car.execute(
                line_tuple=(0.50, DEFAULT_INTERCEPT_OFFSET, 820),
                stop_flags=[],
            )
        print("Terminating the car...")

    def start_main_process(self) -> None:
        threads: List[Thread] = []
        threads.append(Thread(target=self.start_csi_comm))
        threads.append(Thread(target=self.start_rgbd_comm))
        print("Starting the demo...")
        for thread in threads:
            thread.start()
        self.start_car_comm()
        print("Demo started, waiting for threads to join...")
        for thread in threads:
            thread.join()
        print("Demo complete.")