
import time
from typing import Dict, Tuple
from multiprocessing import Queue

import cv2
import numpy as np

from core.qcar import PhysicalCar
from core.control.edge_finder import EdgeFinder, NoContourException
from core.utils.ipc_utils import fetch_latest_in_queue
from core.utils.ipc_utils import put_latest_in_queue
from core.utils.ipc_utils import clear_queue
from core.policies.pid_policy import CompositePIDPolicy
from .exceptions import HaltException
from .observe import DecisionMaker


class ObserverWrapper:
    # TODO: Implement the observer
    def __init__(self) -> None:
        self.observer: DecisionMaker = DecisionMaker(
            classic_traffic_pipeline=True,
            network_class=None,
            output_postprocess=lambda x: x.argmax().item(),
            weights_file=None,
            device='cuda'
        )

    # TODO: Implement the method
    def execute(self, request: Queue, response: Queue) -> None:
        # get the latest image from the multiprocessing queue
        image: np.ndarray = fetch_latest_in_queue(request)
        if image is not None:
            # send the image to the observer
            self.observer(image)
            # get the detection flags
            result: dict = self.observer.detection_flags
            # put the data in the response queue
            put_latest_in_queue(result, response)
            # set watch dog event


class EdgeFinderWrapper:
    def __init__(self, edge_finder: EdgeFinder) -> None:
        self.edge_finder: EdgeFinder = edge_finder

    def execute(self, request: Queue, response: Queue) -> None:
        # get the latest image from the multiprocessing queue
        try:
            start = time.time()
            # print("Getting image from the queue...")
            image: np.ndarray = fetch_latest_in_queue(request)
            if image is not None:
                # get the result from the edge finder
                line: Tuple[float, float] = self.edge_finder.execute(image)
                # preprocess the result to the required format
                result: Tuple[float, float, int] = (line[0], line[1], image.shape[1])
                # mock hardware delay
                end = time.time() - start
                time.sleep(max(0, 0.015 - end)) # thread yielding
                # put the data in the response queue
                put_latest_in_queue(result, response)
                # set watch dog event
        except NoContourException:
            print("No contour!")


class PIDControlCar(PhysicalCar):
    def __init__(self, throttle_coeff: float, steering_coeff: float, desired_speed: float = 1.40) -> None:
        super().__init__(throttle_coeff, steering_coeff)
        self.policy: CompositePIDPolicy = CompositePIDPolicy(expected_velocity=desired_speed)
        self.brake_time: float = (desired_speed - 1.0) / 1.60
        self.last_state: Tuple[float, float, int] = None
        self.action: np.ndarray = np.zeros(2)

    def setup(self, pid_gains: Dict[str, list], offsets: Tuple[float, float]) -> None:
        steering_gains: list = pid_gains['steering']
        throttle_gains: list = pid_gains['throttle']
        self.policy.setup_throttle(k_p=throttle_gains[0], k_i=throttle_gains[1], k_d=throttle_gains[2])
        self.policy.setup_steering(k_p=steering_gains[0], k_i=steering_gains[1], k_d=steering_gains[2], offsets=offsets)
        self.policy.reset_start_time()

    def terminate(self) -> None:
        self.running_gear.terminate()

    def handle_observe(self, observe_queue: Queue) -> None:
        stop_flags: Dict[str, bool] = fetch_latest_in_queue(observe_queue)
        if stop_flags is None: return

        # if stop_flags['red_light']:
        #     halt_time: float = 0.1
        #     raise HaltException(stop_time=halt_time)
        if stop_flags['stop_sign']:
            halt_time: float = 3 + self.brake_time
            raise HaltException(stop_time=halt_time)

    def handle_control(self, edge_queue: Queue):
        # get the observations
        estimated_speed: float = self.estimate_speed()
        # realtime_message_output(f"Estimated speed: {estimated_speed:.2f} m/s")
        edge_info: Tuple[float, float, int] = fetch_latest_in_queue(edge_queue)

        # use last state or new state
        if edge_info is not None: # get valid info from edge finder
            # send observations to the policy
            self.last_state = edge_info
            self.action, _ = self.policy.execute(
                steering_input=edge_info, # slope, intercept, image_width
                linear_speed=estimated_speed # speed obtained from encoder
            )
        elif self.last_state is not None: # use last state
            edge_info = self.last_state
        else:
            return

        # apply action to the car
        self.handle_leds(throttle=self.action[0], steering=self.action[1])
        self.running_gear.read_write_std(
            throttle=self.action[0], steering=self.action[1], LEDs=self.leds
        ) # apply actions to the car

    def execute(self, edge_queue: Queue, observe_queue: Queue) -> None:
        try:
            self.handle_observe(observe_queue)
            self.handle_control(edge_queue)
        except HaltException as e:
            print(f"Stopping the car for {e.stop_time:.2f} seconds")
            self.halt_car(steering=self.action[1], halt_time=e.stop_time)
            self.policy.reset_start_time()
            self.last_state = None # clear the last state
        time.sleep(0.001) # thread yielding
