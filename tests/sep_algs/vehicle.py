import time
import hashlib
from typing import Dict

import cv2
import numpy as np

from core.control.edge_finder import NoContourException
from core.qcar import VirtualCSICamera, PhysicalCar
from core.control.edge_finder import TraditionalEdgeFinder, EdgeFinder
from core.policies.vision_lanefollowing import VisionLaneFollowing, BasePolicy
from core.utils.ipc_utils import StructedDataTypeFactory, SharedMemoryWrapper


class ObserveAlgModule:
    def __init__(self, observe_image_size: np.ndarray) -> None:
        protocol: np.dtype = StructedDataTypeFactory().create_dtype(num_of_cmds=5, image_size=observe_image_size)
        self.memory: SharedMemoryWrapper = SharedMemoryWrapper(protocol, 'observe', True)


class ControlAlgModule:
    def __init__(self, control_image_size: np.ndarray) -> None:
        protocol: np.dtype = StructedDataTypeFactory().create_dtype(num_of_cmds=4, image_size=control_image_size)
        self.memory: SharedMemoryWrapper = SharedMemoryWrapper(protocol, 'control', True)
        self.last_timestamp: float = -1.0

    def setup(self, expected_velocity: float, pid_gains: Dict[str, list]) -> None:
        steering_gains: list = pid_gains['steering']
        throttle_gains: list = pid_gains['throttle']
        edge_finder: EdgeFinder = TraditionalEdgeFinder(image_width=820, image_height=410)
        self.policy: BasePolicy = VisionLaneFollowing(edge_finder=edge_finder, expected_velocity=expected_velocity)
        self.policy.setup_steering(k_p=steering_gains[0], k_i=steering_gains[1], k_d=steering_gains[2])
        self.policy.setup_throttle(k_p=throttle_gains[0], k_i=throttle_gains[1], k_d=throttle_gains[2])

    def read_data(self) -> None:
        data_and_command: np.ndarray = self.memory.read_from_shm('data_and_commands')
        if data_and_command[0] < 0:
            self.policy.reset_start_time()
        self.estimated_speed: float = data_and_command[1]
        timestamp: float = self.memory.read_from_shm('timestamp')
        if timestamp == self.last_timestamp:
            return
        self.last_timestamp = timestamp # update timestamp
        self.image: np.ndarray = self.memory.read_from_shm('image').copy()

    def execute(self, lock) -> None:
        try:
            with lock:
                self.read_data()
                action, _ = self.policy.execute(self.image, self.estimated_speed, 1.0)
                command: np.ndarray = np.zeros(2) # clear reset, estimated speed
                self.memory.write_to_shm('data_and_commands', np.concatenate([command, action]))
        except NoContourException:
            pass


class HardwareModule(PhysicalCar):
    def __init__(self, throttle_coeff: float = 0.3, steering_coeff: float = 0.5, desired_speed: float = 0) -> None:
        super().__init__(throttle_coeff, steering_coeff)
        self.front_csi: VirtualCSICamera = VirtualCSICamera(id=3)
        self.brake_time: float = (desired_speed - 1.0) / 1.60
        self.action: np.ndarray = np.zeros(2)
        self.momories: Dict[str, SharedMemoryWrapper] = {}

    def setup(self, control_image_size: tuple, observe_image_size: tuple) -> None:
        control_protocol = StructedDataTypeFactory().create_dtype(num_of_cmds=4, image_size=control_image_size)
        observe_protocol = StructedDataTypeFactory().create_dtype(num_of_cmds=5, image_size=observe_image_size)
        self.momories['control'] = SharedMemoryWrapper(control_protocol, 'control', False)
        # self.momories['observe'] = SharedMemoryWrapper(observe_protocol, 'observe', False)

    def terminate(self) -> None:
        self.running_gear.terminate()
        self.front_csi.terminate()

    def handle_event(self) -> float:
        return 1.0

    def transmit_data(self, locks, shm_name, image_data: np.ndarray, data_and_command: np.ndarray) -> None:
        with locks[shm_name]:
            current_time: float = time.time()
            hash_code: float = hashlib.sha256(str(current_time).encode()).hexdigest()
            self.momories[shm_name].write_to_shm('timestamp', float.fromhex(hash_code[:16]))
            self.momories[shm_name].write_to_shm('image', image_data)
            if data_and_command is None:
                return
            self.momories[shm_name].write_to_shm('data_and_commands', data_and_command)

    def read_action(self, lock) -> np.ndarray:
        with lock:
            return self.momories['control'].read_from_shm('data_and_commands')[2:]

    def execute(self, locks: dict) -> None:
        # handle the stop events
        self.handle_event()
        # estimate the current speed
        current_speed: float = self.estimate_speed()
        # transmit data to other processes
        front_image: np.ndarray = self.front_csi.read_image()
        if front_image is None:
            return
        # transmit data to algorithm modules
        self.transmit_data(locks, 'control', front_image, np.array([1.0, current_speed, 0.0, 0.0]))
        # self.transmit_data(locks, 'observe', front_image, None)
        # get the action from the control algorithm modules
        self.action: np.ndarray = self.read_action(locks['control'])
        # execute the action
        self.running_gear.read_write_std(throttle=self.action[0], steering=self.action[1], LEDs=self.leds)
