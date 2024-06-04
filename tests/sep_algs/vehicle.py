import time
import hashlib
from typing import Dict, Tuple

import cv2
import numpy as np

from core.control.edge_finder import NoContourException
from core.qcar import VirtualCSICamera, VirtualRGBDCamera, PhysicalCar
from core.control.edge_finder import TraditionalEdgeFinder, EdgeFinder
from core.policies.vision_lanefollowing import VisionLaneFollowing, BasePolicy
from core.utils.tools import realtime_message_output
from core.utils.ipc_utils import StructedDataTypeFactory, SharedMemoryWrapper
from .observe import DecisionMaker
from .exceptions import HaltException


class ObserveAlgModule:
    def __init__(self, observe_image_size: np.ndarray) -> None:
        protocol: np.dtype = StructedDataTypeFactory().create_dtype(num_of_cmds=4, image_size=observe_image_size)
        self.memory: SharedMemoryWrapper = SharedMemoryWrapper(protocol, 'observe', True)
        self.last_timestamp: float = None
        self.observer: DecisionMaker = DecisionMaker(
            classic_traffic_pipeline=True,
            network_class=None,
            output_postprocess=lambda x: x.argmax().item(),
            weights_file=None,
            device='cuda'
        )

    def read_data(self) -> None:
        timestamp: float = self.memory.read_from_shm('timestamp')
        will_read_image: bool = timestamp != self.last_timestamp
        if will_read_image:
            self.last_timestamp = timestamp # update timestamp
            self.image: np.ndarray = self.memory.read_from_shm('image').copy()
        return will_read_image

    def execute(self, lock) -> None:
        with lock:
            if not self.read_data():
                return

            self.observer(self.image)
            detection_flags: dict = self.observer.detection_flags
            # realtime_message_output(f"Detection flags: {detection_flags}")
            transmit_data: np.ndarray = np.array([
                1.0 if detection_flags['stop_sign'] else -1.0,
                1.0 if detection_flags['horizontal_line'] else -1.0,
                1.0 if detection_flags['red_light'] else -1.0,
                1.0 if detection_flags['unknown_error'] else -1.0,
            ])
            self.memory.write_to_shm('data_and_commands', transmit_data)
            #print(transmit_data)


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

    def read_data(self) -> bool:
        timestamp: float = self.memory.read_from_shm('timestamp')
        will_read_image: bool = timestamp != self.last_timestamp
        if will_read_image:
            self.last_timestamp = timestamp # update timestamp
            self.image: np.ndarray = self.memory.read_from_shm('image').copy()

            data_and_command: np.ndarray = self.memory.read_from_shm('data_and_commands')
            self.estimated_speed: float = data_and_command[1]
            if data_and_command[0] > 0:
                self.policy.reset_start_time()
        return will_read_image

    def execute(self, lock) -> None:
        try:
            with lock:
                if not self.read_data():
                    return
                action, _ = self.policy.execute(self.image, self.estimated_speed, 1.0)
                command: np.ndarray = np.array([0.0, 0.0]) # clear reset, estimated speed
                self.memory.write_to_shm('data_and_commands', np.concatenate([command, action]))
        except NoContourException:
            pass
            # print('No contour detected')


class HardwareModule(PhysicalCar):
    def __init__(self, throttle_coeff: float = 0.3, steering_coeff: float = 0.5, desired_speed: float = 0) -> None:
        super().__init__(throttle_coeff, steering_coeff)
        self.front_csi: VirtualCSICamera = VirtualCSICamera(id=3)
        self.rgbd_camera: VirtualRGBDCamera = VirtualRGBDCamera()
        self.brake_time: float = (desired_speed - 1.0) / 1.60 if desired_speed > 1.0 else 0.0
        self.memories: Dict[str, SharedMemoryWrapper] = {}
        self.action: np.ndarray = np.zeros(2)
        self.reset: float = 1.0

    def setup(self, control_image_size: tuple, observe_image_size: tuple) -> None:
        control_protocol = StructedDataTypeFactory().create_dtype(num_of_cmds=4, image_size=control_image_size)
        observe_protocol = StructedDataTypeFactory().create_dtype(num_of_cmds=4, image_size=observe_image_size)
        self.memories['control'] = SharedMemoryWrapper(control_protocol, 'control', False)
        self.memories['observe'] = SharedMemoryWrapper(observe_protocol, 'observe', False)

    def terminate(self) -> None:
        self.running_gear.terminate()
        self.front_csi.terminate()

    # TODO: Implement this method
    def read_halt_event(self, lock) -> None:
        with lock:
            flags: np.ndarray = self.memories['observe'].read_from_shm('data_and_commands')
        if flags[0] > 0:
            # print('Stop sign detected')
            halt_time: float = 3 + self.brake_time
            raise HaltException(stop_time=halt_time)
        # elif flags[2] > 0:
        #     # print('Red light detected')
        #     halt_time: float = 0.1
        #     raise HaltException(stop_time=halt_time)
        else:
            self.reset = 0.0

    def transmit_data(self, locks, shm_name, image_data: np.ndarray, data_and_command: np.ndarray) -> None:
        with locks[shm_name]:
            if image_data is not None:
                self.memories[shm_name].write_to_shm('timestamp', time.time())
                self.memories[shm_name].write_to_shm('image', image_data)
            if data_and_command is not None:
                self.memories[shm_name].write_to_shm('data_and_commands', data_and_command)

    def read_action(self, lock) -> np.ndarray:
        with lock:
            return self.memories['control'].read_from_shm('data_and_commands')[2:]

    def execute(self, locks: dict) -> None:
        try:
            rgbd_image: np.ndarray = self.rgbd_camera.read_rgb_image()
            self.transmit_data(locks, 'observe', rgbd_image, None) # no need to transmit command here
            self.read_halt_event(locks['observe'])

            front_image: np.ndarray = self.front_csi.read_image()
            current_speed: float = self.estimate_speed()
            self.transmit_data(locks, 'control', front_image, np.concatenate(([self.reset, current_speed], self.action)))
            self.action: np.ndarray = self.read_action(locks['control'])
            self.running_gear.read_write_std(throttle=self.action[0], steering=self.action[1], LEDs=self.leds)
        except HaltException as e:
            self.reset = 1.0
            self.halt_car(steering=self.action[1], halt_time=e.stop_time)
