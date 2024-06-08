import time
import hashlib
from typing import Dict, Tuple

import cv2
import numpy as np

from core.control.edge_finder import NoContourException, NoImageException
from core.qcar import VirtualCSICamera, VirtualRGBDCamera, PhysicalCar
from core.control.edge_finder import TraditionalEdgeFinder, EdgeFinder
from core.policies.vision_lanefollowing import VisionLaneFollowing, BasePolicy
from core.utils.performance import mock_delay, skip_delay
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

    # TODO: Move this method to a concrete SharedMemoryWrapper class？
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
    def __init__(self, control_image_size: np.ndarray, will_mock_delay: bool = False) -> None:
        protocol: np.dtype = StructedDataTypeFactory().create_dtype(num_of_cmds=4, image_size=control_image_size)
        self.memory: SharedMemoryWrapper = SharedMemoryWrapper(protocol, 'control', True)
        self.last_timestamp: float = -1.0
        if will_mock_delay:
            self.execute_delay = mock_delay
        else:
            self.execute_delay = skip_delay

    def setup(self, expected_velocity: float, pid_gains: Dict[str, list], offsets: Tuple[float, float]) -> None:
        steering_gains: list = pid_gains['steering']
        throttle_gains: list = pid_gains['throttle']
        edge_finder: EdgeFinder = TraditionalEdgeFinder(image_width=820, image_height=410)
        self.policy: BasePolicy = VisionLaneFollowing(edge_finder=edge_finder, expected_velocity=expected_velocity)
        self.policy.setup_steering(k_p=steering_gains[0], k_i=steering_gains[1], k_d=steering_gains[2], offsets=offsets)
        self.policy.setup_throttle(k_p=throttle_gains[0], k_i=throttle_gains[1], k_d=throttle_gains[2])

    # TODO: Move this method to a concrete SharedMemoryWrapper class？
    def read_data(self) -> bool:
        timestamp: float = self.memory.read_from_shm('timestamp')
        data_and_command: np.ndarray = self.memory.read_from_shm('data_and_commands')
        will_read_image: bool = timestamp != self.last_timestamp
        if will_read_image:
            self.last_timestamp = timestamp # update timestamp
            self.image: np.ndarray = self.memory.read_from_shm('image').copy()

        self.estimated_speed: float = data_and_command[1]
        # print(f"Reset Flag: {data_and_command[0]:.4f}")
        if data_and_command[0] > 0:
            # print("Resetting the start time of the controllers")
            self.policy.reset_start_time()
        return will_read_image

    def execute(self, lock) -> None:
        try:
            start: float = time.time()
            with lock:
                if not self.read_data():
                    return
                action, _ = self.policy.execute(self.image, self.estimated_speed, 1.0)
                command: np.ndarray = np.array([0.0, 0.0]) # clear reset, estimated speed
                self.execute_delay(start, 0.0125) # mock hardware delay
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
        # self.writer: ImageWriter = ImageWriter('images')


    def setup(self, control_image_size: tuple, observe_image_size: tuple) -> None:
        control_protocol = StructedDataTypeFactory().create_dtype(num_of_cmds=4, image_size=control_image_size)
        observe_protocol = StructedDataTypeFactory().create_dtype(num_of_cmds=4, image_size=observe_image_size)
        self.memories['control'] = SharedMemoryWrapper(control_protocol, 'control', False)
        self.memories['observe'] = SharedMemoryWrapper(observe_protocol, 'observe', False)

    def terminate(self) -> None:
        self.running_gear.terminate()
        self.front_csi.terminate()

    # TODO: Implement this method
    def handle_halt_event(self, lock) -> None:
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

    # TODO: Move this method to a concrete SharedMemoryWrapper class？
    def transmit_data(self, locks, shm_name, image_data: np.ndarray, data_and_command: np.ndarray) -> None:
        with locks[shm_name]:
            if image_data is not None:
                self.memories[shm_name].write_to_shm('timestamp', time.time())
                self.memories[shm_name].write_to_shm('image', image_data)
            if data_and_command is not None:
                self.memories[shm_name].write_to_shm('data_and_commands', data_and_command)

    # TODO: Move this method to a concrete SharedMemoryWrapper class？
    def read_action(self, lock) -> np.ndarray:
        with lock:
            return self.memories['control'].read_from_shm('data_and_commands')[2:]

    def execute(self, locks: dict) -> None:
        try:
            rgbd_image: np.ndarray = self.rgbd_camera.read_rgb_image()
            self.transmit_data(locks, 'observe', rgbd_image, None) # no need to transmit command here
            self.handle_halt_event(locks['observe'])

            front_image: np.ndarray = self.front_csi.read_image()
            current_speed: float = self.estimate_speed()
            self.transmit_data(locks, 'control', front_image, np.concatenate(([0.0, current_speed], self.action)))
            self.action = self.read_action(locks['control'])
            # self.handle_leds(throttle=self.action[0], steering=self.action[1])
            self.running_gear.read_write_std(throttle=self.action[0], steering=self.action[1], LEDs=self.leds)
            # if front_image is not None:
            #     self.writer.add_image(front_image.copy())
        except HaltException as e:
            print(f"Stopping the car for {e.stop_time:.2f} seconds")
            self.memories['control'].write_to_shm('data_and_commands', np.concatenate(([1.0, 0.0], self.action)))
            self.halt_car(steering=self.action[1], halt_time=e.stop_time)
