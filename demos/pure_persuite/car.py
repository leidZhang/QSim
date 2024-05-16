import time
import sys
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from qvl.qlabs import QuanserInteractiveLabs
from pal.products.qcar import QCar

from core.environment.exception import AnomalousEpisodeException
from core.sensor.sensor import VirtualCSICamera
from core.qcar.monitor import Monitor
from core.policies.pure_persuit import PurePursuitPolicy


class PPMonitor:
    def __init__(
        self,
        qlabs: QuanserInteractiveLabs,
        waypoint_sequence: np.ndarray
    ) -> None:
        self.qlabs: QuanserInteractiveLabs = qlabs
        self.waypoint_sequence: np.ndarray = waypoint_sequence
        self.observation: dict = {}
        self.monitor: Monitor = Monitor(160, 0, 0.05)

    def get_state(self) -> tuple:
        self.monitor.get_state(self.qlabs)
        ego_state: np.ndarray = self.monitor.state
        orig: np.ndarray = ego_state[:2]
        yaw: float = -ego_state[2]
        rot: np.ndarray = np.array([
            [np.cos(yaw), np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)]
        ])
        return orig, yaw, rot

    def setup(self) -> np.ndarray:
        orig, yaw, rot = self.get_state()
        self.current_waypoint_index: int = 0
        self.next_waypoints: np.ndarray = self.waypoint_sequence
        self.observation["waypoints"] = np.matmul(self.next_waypoints[:200] - orig, rot)
        return orig

    def update(self) -> None:
        orig, yaw, rot = self.get_state()
        waypoints: np.ndarray = np.roll(
            self.waypoint_sequence,
            -self.current_waypoint_index,
            axis=0)[:200]
        norm_dist: np.ndarray = np.linalg.norm(waypoints - orig, axis=1)
        dist_ix: int = np.argmin(norm_dist)
        self.current_waypoint_index = (self.current_waypoint_index + dist_ix) % self.waypoint_sequence.shape[0]
        self.next_waypoints = self.next_waypoints[dist_ix:]  # clear pasted waypoints
        self.observation['waypoints'] = np.matmul(self.next_waypoints[:200] - orig, rot)


class PPCar:
    def __init__(
        self, qlabs: QuanserInteractiveLabs,
        waypoint_sequence: np.ndarray
    ) -> None:
        self.monitor: PPMonitor = PPMonitor(qlabs, waypoint_sequence)
        self.policy: PurePursuitPolicy = PurePursuitPolicy(0.5)
        self.running_gear: QCar = QCar()

    def setup(self) -> None:
        self.start_orig = self.monitor.setup()

    def halt_car(self) -> None:
        self.running_gear.read_write_std(0, 0)

    def execute(self) -> None:
        observation = self.monitor.observation
        action, _ = self.policy(observation)
        self.throttle: float = 0.08 * action[0]
        self.steering: float = 0.5 * action[1]
        self.running_gear.read_write_std(throttle=self.throttle, steering=self.steering)
        self.monitor.update()


class MPCar:
    def transmit_action(self, action: np.ndarray, shm_name: str, step_event) -> None:
        if not step_event.is_set():
            shm = SharedMemory(name=shm_name)
            data_shared: np.ndarray = np.ndarray((2,), dtype=np.float64, buffer=shm.buf)
            np.copyto(data_shared, action)
            step_event.set()
            shm.close()


class PPCarMP(PPCar, MPCar):
    def execute(self, shm_name, step_event) -> None:
        super().execute()
        action: np.ndarray = np.array([self.throttle, self.steering])
        self.transmit_action(action, shm_name, step_event)


class PPCarCan(PPCar, MPCar): 
    def __init__(self, qlabs: QuanserInteractiveLabs, waypoint_sequence: np.ndarray) -> None:
        super().__init__(qlabs, waypoint_sequence)
        self.front_csi = VirtualCSICamera()

    def transmit_action(
            self, action: np.ndarray, 
            image: np.ndarray, 
            shm_name_image: str, 
            shm_name_action: str, 
            step_event
        ) -> None:
        if not step_event.is_set():
            shm_action: SharedMemory = SharedMemory(name=shm_name_action)
            shm_image: SharedMemory = SharedMemory(name=shm_name_image)
            action_shared: np.ndarray = np.ndarray((2,), dtype=np.float64, buffer=shm_action.buf)
            image_shared: np.ndarray = np.ndarray((410, 820, 3), dtype=np.uint8, buffer=shm_image.buf)
            np.copyto(action_shared, action)
            np.copyto(image_shared, image)
            step_event.set()
            shm_action.close()
            shm_image.close()

    def execute(self, shm_name_image: str, shm_name_action: str, step_event) -> None:
        super().execute()
        action: np.ndarray = np.array([self.throttle, self.steering])
        image: np.ndarray = self.front_csi.await_image()
        self.transmit_action(action, image, shm_name_image, shm_name_action, step_event)