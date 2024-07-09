import os
import time
from queue import Queue
from typing import Union
from multiprocessing import Queue as MPQueue

import cv2
import numpy as np

from core.utils.executions import BaseThreadExec, BaseProcessExec
from core.utils.ipc_utils import put_latest_in_queue
from core.utils.performance import skip
from core.qcar import VirtualCSICamera
from core.qcar import LidarSLAM
from tests.performance_environment import prepare_test_environment
from .utils import correct_traj, prepare_task
# from settings import TASK, DEFAULT_PWM, DESIRED_SPEED
from .modules import WaypointProcessor, ResNetWrapper
from .modules import PurepursuitCar, PhysicalCar

TASK = [10, 14]
DEFAULT_PWM = 0.08

CALIBRATION_POSE = [0,2,-np.pi / 2]


class ResNetDetectorExec(BaseProcessExec):
    def create_instance(self) -> ResNetWrapper:
        return ResNetWrapper()


class WaypointProcessorExec(BaseProcessExec):
    def create_instance(self, request_queue: MPQueue) -> WaypointProcessor:
        waypoints: np.ndarray = prepare_test_environment(node_id=10)
        # waypoints: np.ndarray = prepare_task(TASK)
        # waypoints = correct_traj(waypoints)
        waypoint_processor: WaypointProcessor = WaypointProcessor(waypoints=waypoints)
        waypoint_processor.setup(0, request_queue)
        return waypoint_processor
        
    def run_process(self, request_queue: MPQueue, response_queue: MPQueue) -> None:
        instance: WaypointProcessor = self.create_instance(request_queue)
        while not self.done.is_set():
            instance.execute(request_queue, response_queue)
        self.final()

class ObservationComm(BaseThreadExec):
    def __init__(self, debug: bool = False) -> None:
        super().__init__()
        if debug:
            self.start_time = time.time()
            self.final_operation = self.__debug
        else:
            self.final_operation = skip

    def __debug(self, image: np.ndarray) -> None:
        if time.time() - self.start_time < 5:
            return

        if image is not None:
            cv2.imshow("CSI image", image)
            cv2.waitKey(1)

    def setup_thread(self) -> None:
        self.camera: VirtualCSICamera = VirtualCSICamera(id=3)
        self.gps: LidarSLAM = LidarSLAM(CALIBRATION_POSE, False)

    def terminate(self) -> None:
        super().terminate()
        self.camera.terminate()
        self.gps.terminate()

    def execute(
        self, 
        detection_response_queue: MPQueue, 
        gps_response_queue: MPQueue,
        obstacle_response_queue: MPQueue,
    ) -> None:
        # read lidar slam 
        self.gps.readGPS()
        state: np.ndarray = np.array([
            self.gps.position[0],
            self.gps.position[1],
            self.gps.orientation[2],
            0, 0, 0
        ])
        # read csi camera
        image: np.ndarray = self.camera.read_image()
        if image is not None:
            put_latest_in_queue(image.copy(), detection_response_queue)
            put_latest_in_queue(image.copy(), obstacle_response_queue)
        put_latest_in_queue(state.copy(), gps_response_queue)
        self.final_operation(image)
        
    def final_execute(self) -> None:
        self.terminate()


class PurepursuitCarComm(BaseThreadExec):
    def __init__(self, waypoints: np.ndarray = None) -> None:
        super().__init__()
        self.waypoints: np.ndarray = waypoints

    def setup_thread(self) -> None:
        self.car: PhysicalCar = PurepursuitCar(desired_speed=DEFAULT_PWM)
        self.car.setup()

    def terminate(self) -> None:
        super().terminate()
        self.car.halt_car()
        self.car.running_gear.terminate()

    def execute(
        self, 
        detection_request_queue: MPQueue, 
        observation_request_queue: MPQueue,
        obstacle_request_queue: MPQueue
    ) -> None:
        self.car.execute(detection_request_queue, observation_request_queue, obstacle_request_queue)

    # def run_thread(self, detection_request_queue: MPQueue, observation_request_queue: Queue) -> None:
    #     # prepare the thread instances
    #     self.setup_thread(observation_request_queue)
    #     # setup the watchdog
    #     self.reach_new_stage()
    #     # main loop of the thread
    #     while not self.done.is_set():
    #         self.execute(detection_request_queue, observation_request_queue)
    #         # set the watchdog
    #         self.reach_new_stage()
    #     # final execution to close resources
    #     self.final_execute()

    def final_execute(self) -> None:
        self.terminate()