from multiprocessing.shared_memory import SharedMemory

import numpy as np

from qvl.qlabs import QuanserInteractiveLabs
from pal.products.qcar import QCar

from core.simulator.monitor import Monitor
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
        ego_state = self.monitor.state
        orig: np.ndarray = ego_state[:2]
        yaw: float = -ego_state[2]
        rot: np.ndarray = np.array([
            [np.cos(yaw), np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)]
        ])
        return orig, yaw, rot

    def setup(self) -> None:
        orig, yaw, rot = self.get_state()
        self.current_waypoint_index: int = 0
        self.next_waypoints: np.ndarray = self.waypoint_sequence
        self.observation["waypoints"] = np.matmul(self.next_waypoints[:200] - orig, rot)

    def update(self) -> np.ndarray:
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
        self.monitor.setup()

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
    def transmit_action(action: np.ndarray, shm_name: str, lock):
        with lock:
            shm = SharedMemory(name=shm_name)
            action_shared: np.ndarray = np.ndarray(action.shape, dtype=action.dtype, buffer=shm.buf)
            action_shared[:] = action[:]
            shm.close()

class PPCarMP(PPCar, MPCar):
    def execute(self, shm_name) -> None:
        super().execute()
        action: np.ndarray = np.array([self.throttle, self.steering])
        self.transmit_action(action, shm_name)
