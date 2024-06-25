import time
import base64
from queue import Queue
from datetime import datetime
from typing import Tuple, List
from multiprocessing import Queue as MPQueue

import cv2
import numpy as np
from qvl.qlabs import QuanserInteractiveLabs

from core.policies.pure_persuit import PurePursuiteAdaptor
from core.policies.base_policy import BasePolicy
from core.qcar.monitor import Monitor
from core.qcar.vehicle import PhysicalCar
from core.qcar.sensor import VirtualCSICamera
from core.qcar.constants import QCAR_ACTOR_ID
from core.datatypes.pose import MockPose
from core.utils.io_utils import DataWriter, JSONDataWriter, NPZDataWriter
from core.utils.control_utils import get_yaw_noise
from core.utils.performance import elapsed_time, realtime_message_output
from constants import MAX_LOOKAHEAD_INDICES
from .reward_funcs import reward_based_on_indices_and_noise
from .exceptions import ReachGoalException

EPISODE_KEYS: List[str] = ["timestamp", "task", "task_length"]
ARRAY_KEYS: List[str] = ["state", "action", "noise"]


class RecordDataWriter:
    def __init__(self, folder_path: str, stop_event) -> None:
        self.stop_event = stop_event
        self.data_writer: DataWriter = JSONDataWriter(folder_path)
        self.data_writer.process_data = self.preprocess
        self.last_timestamp: str = None
        self.last_task_length: int = 0

    def _convert_data_format(self, data: dict) -> None:
        # convert the waypoints to base64 since it is a large array
        data["waypoints"] = base64.encodebytes(data["waypoints"].tobytes()).decode("utf-8")
        # convert the state, action, and noise to float since they are small arrays
        for key in ARRAY_KEYS:
            converted_data: list = []
            for i in range(len(data[key])):
                converted_data.append(float(data[key][i]))
            data[key] = converted_data

    def _delete_keys(self, data: dict) -> None:
        for key in EPISODE_KEYS:
            data.pop(key, None)
        data.pop("last_index", None)

    def _write_image(self, data: dict, folder_path: str) -> None:
        # Generate a unique name for the image based on the episode timestamp
        image_path: str = f"image_{data['timestamp']}_{data['id']}.jpg"
        abs_image_path: str = f"{self.data_writer.folder_path}/{folder_path}/{image_path}"
        # Write the image to the output path
        cv2.imwrite(abs_image_path, data["front_csi_image"])
        # replace the image with the image path
        data["front_csi_image"] = image_path

    def cal_required_data(self, data: dict, index: int) -> dict:
        data["id"] = index
        # calculate the reward based on the client's requirements
        data["reward"] = reward_based_on_indices_and_noise(
            prev_pos=data["last_index"],
            cur_pos=data["current_index"],
            noise=data["noise"],
            last_task_len=self.last_task_length,
        )
        # correct the index if it is negative
        if data["current_index"] < 0:
            data["current_index"] = self.last_task_length + data["current_index"]

    def preprocess(self, data: dict, index: int, folder_path: str) -> None:
        self.cal_required_data(data, index)
        self._write_image(data, folder_path)
        self._convert_data_format(data)
        self._delete_keys(data)

    def execute(self, data_queue: MPQueue) -> None:
        if data_queue.empty():
            return

        data: dict = data_queue.get()
        if self.last_timestamp is None:
            self.last_timestamp = data["timestamp"]

        if data["timestamp"] != self.last_timestamp and data["current_index"] >= 0:
            print(f"Current buffer size: {len(self.data_writer.history)}")
            self.last_task_length: int = self.data_writer.history[0]["task_length"]
            self.last_timestamp = data["timestamp"]
            self.data_writer.write_data(data["timestamp"])
            # if len(self.data_writer.history) > 50_000:
            #     self.data_writer.history.set()
        # add to the buffer
        self.data_writer.add_data(data)


class MockOptitrackClient:
    def __init__(self, data_queue: Queue, actor_number: int=0) -> None:
        self.monitor: Monitor = Monitor(class_id=QCAR_ACTOR_ID, actor_number=actor_number, dt=0.01)
        self.data_queue: Queue = data_queue

    def execute(self, qlabs: QuanserInteractiveLabs) -> None:
        # get the state from the monitor
        self.monitor.read_state(qlabs)
        ego_state: np.ndarray = self.monitor.state
        # convert the state to a mock pose object
        pose: MockPose = MockPose(ego_state)
        # put the pose object to the data queue
        if self.data_queue.full():
            self.data_queue.get()
        self.data_queue.put(pose)


class PurePursuiteCar(PhysicalCar): # for simulation purpose
    def __init__(
        self, 
        stop_event,
        qlabs: QuanserInteractiveLabs,
        throttle_coeff: float = 0.3, 
        steering_coeff: float = 0.5,
    ) -> None:
        super().__init__(throttle_coeff, steering_coeff)
        # initialize the variables
        self.stop_event = stop_event
        self.start_time: float = None
        self.linear_speed: float = 0.0
        self.task_start_index: int = 0
        self.last_waypoint_index: int = 0
        self.noise: np.ndarray = np.zeros(2)
        self.action: np.ndarray = np.zeros(2)
        self.observation: dict = {}
        self.data_queue: Queue = Queue(5)
        # initialize the car components
        self.qlabs: QuanserInteractiveLabs = qlabs
        self.front_csi: VirtualCSICamera = VirtualCSICamera(id=3)
        self.client: MockOptitrackClient = MockOptitrackClient(self.data_queue)
        self.policy: BasePolicy = PurePursuiteAdaptor()
        # self.writer: RecordDataWriter = RecordDataWriter("output", "data.json")
        
    def get_ego_state(self) -> np.ndarray:
        self.client.execute(self.qlabs) # mock client thread operation
        pose: MockPose = self.data_queue.get()
        state: np.ndarray = np.array([
            pose.position_x, 
            pose.position_y,
            pose.orientation,
            pose.velocity_x,
            pose.velocity_y,
            pose.angular_speed
        ]) # x, y, theta, v_x, v_y, omega
        return state
    
    def cal_vehicle_state(self, ego_state: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        orig: np.ndarray = ego_state[:2]
        yaw: float = -ego_state[2]
        # print(f"yaw: {yaw}")
        rot: np.ndarray = np.array([
            [np.cos(yaw), np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)]
        ])
        return orig, yaw, rot

    def handle_observation(self, orig: np.ndarray, rot: np.ndarray, image: np.ndarray) -> None:
        # in case of less than 200 waypoints
        if self.next_waypoints.shape[0] < MAX_LOOKAHEAD_INDICES:
            slop = MAX_LOOKAHEAD_INDICES - self.next_waypoints.shape[0]
            self.next_waypoints = np.concatenate([self.next_waypoints, self.waypoints[:slop]])
        # add infos to observation
        self.observation['duration'] = time.time() - self.start_time if self.start_time is not None else 0.0
        self.observation['waypoints'] = np.matmul(self.next_waypoints[:MAX_LOOKAHEAD_INDICES] - orig, rot)
        self.observation['state'] = self.ego_state
        self.observation['action'] = self.action
        self.observation['noise'] = self.noise
        self.observation['timestamp'] = self.episode_timestamp
        self.observation['task'] = self.task
        self.observation['current_index'] = int(self.task_start_index + self.current_waypoint_index)
        self.observation['motor_tach'] = self.linear_speed
        self.observation['task_length'] = int(self.task_length)
        self.observation['last_index'] = int(self.last_waypoint_index)
        self.observation['front_csi_image'] = image.copy() if image is not None else np.zeros((410, 820, 3))
        cv2.imshow('Front CSI Image', self.observation['front_csi_image'])
        self.start_time: float = time.time()

    def setup(self, node_sequence: List[int], waypoints: np.ndarray, init_waypoint_index: int = 0) -> None:
        self.waypoints: np.ndarray = waypoints
        self.ego_state: np.ndarray = self.get_ego_state()
        orig, _, rot = self.cal_vehicle_state(self.ego_state)
        self.task_length: int = len(self.waypoints)
        self.current_waypoint_index: int = init_waypoint_index
        self.next_waypoints: np.ndarray = self.waypoints[self.current_waypoint_index:]
        self.episode_timestamp: str = datetime.now().strftime('%Y%m%d%H%M%S%f')
        self.task: List[int] = node_sequence # task node sequence
        print(f"Executing task {self.task}, task length: {len(self.waypoints)}")
        self.handle_observation(orig, rot, None)

    def update_state(self, image: np.ndarray) -> None:
        # get the ego state from the qlabs
        self.ego_state = self.get_ego_state() 
        # get the original and rotation matrix
        orig, _, rot = self.cal_vehicle_state(self.ego_state)
        # get the local waypoints
        local_waypoints: np.ndarray = np.roll(
            self.waypoints, -self.current_waypoint_index, axis=0
        )[:MAX_LOOKAHEAD_INDICES]
        # get the distance to the waypoints
        self.norm_dist: np.ndarray = np.linalg.norm(local_waypoints - orig, axis=1)
        # get the index of the closest waypoint
        self.dist_ix: int = np.argmin(self.norm_dist)
        # update the current waypoint index
        self.current_waypoint_index = (self.current_waypoint_index + self.dist_ix) % self.waypoints.shape[0]
        # clear pasted waypoints
        self.next_waypoints = self.next_waypoints[self.dist_ix:] 
        # add waypoint to the observation
        self.handle_observation(orig, rot, image)

    def check_new_tasks(self, task_queue: MPQueue) -> None:
        # update the last waypoint index before checking the new tasks
        self.last_waypoint_index = self.task_start_index + self.current_waypoint_index
        if self.current_waypoint_index < len(self.waypoints) - 100:
            return

        if task_queue is not None and not task_queue.empty():
            task_data: Tuple[List[int], np.ndarray] = task_queue.get()
            self.task: List[int] = task_data[0]
            # print(f"Get the new task {task_data[0]}, task length: {len(task_data[1])}") # get the new task
            self.next_waypoints = np.concatenate([self.waypoints[self.current_waypoint_index:], task_data[1]])
            self.task_length = len(task_data[1])
            realtime_message_output(f"Receiving data: {self.task_length}    ")
            self.task_start_index = self.task_length - len(self.next_waypoints)
            self.episode_timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
            self.waypoints = self.next_waypoints # update the waypoints
            self.current_waypoint_index = 0 # reset the current waypoint index
            self.task = task_data[0] # update the task
            print(f"Executing task {self.task}, task length: {self.task_length}")
        else: # we will stop the car if there is no new task
            raise ReachGoalException("Reached the goal")
        
    def handle_data_transmit(self, obs_queue: MPQueue = None) -> None:
        # if self.stop_event.is_set():
        #     self.running_gear.halt_car()
        #     time.sleep(1000) # wait for the data writer to finish
        #     raise ReachGoalException("Reached the goal")

        # transmit data to the data writer process
        if obs_queue.full():
            obs_queue.get()
        # use shallow copy to avoid the data being modified
        obs_queue.put(self.observation.copy())

    def execute(self, task_queue: MPQueue = None, obs_queue: MPQueue = None) -> None:
        image: np.ndarray = self.front_csi.read_image()
        if image is not None:
            # calculate the estimated speed
            self.linear_speed = self.estimate_speed()
            # get the control action from the policy
            action, _ = self.policy.execute(obs=self.observation)
            # execute the control
            self.noise = np.array([0.0, get_yaw_noise(action)])
            self.action[0] = action[0] * self.throttle_coeff
            self.action[1] = action[1] * self.steering_coeff
            self.running_gear.read_write_std(
                throttle=self.action[0] + self.noise[0], 
                steering=self.action[1] + self.noise[1]
            )

            # polling for the new tasks
            self.check_new_tasks(task_queue)

            # update the car state
            self.update_state(image) 
            # transmit the data to the data writer
            self.handle_data_transmit(obs_queue)
            cv2.waitKey(1)
