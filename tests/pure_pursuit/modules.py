import time
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
from core.qcar.constants import QCAR_ACTOR_ID
from core.datatypes.pose import MockPose
from core.utils.io_utils import DataWriter
from core.utils.control_utils import get_yaw_noise
from core.utils.performance import elapsed_time, realtime_message_output
from constants import MAX_LOOKAHEAD_INDICES
from .exceptions import ReachGoalException


class RecordDataWriter:
    def __init__(self, folder_path: str, json_path: str) -> None:
        self.data_writer: DataWriter = DataWriter(folder_path, json_path)
        self.data_writer.process_data = self.preprocess
        self.last_timestamp: str = None

    def _convert_data_format(self, data: dict) -> None:
        data["state"] = data["state"].tolist()
        data["action"] = data["action"].tolist()
        data["waypoints"] = data["waypoints"].tolist()
        data["noise"] = data["noise"].tolist()

    def _write_image(self, data: dict) -> None:
        # Generate a unique name for the image based on the episode timestamp
        image_path: str = f"image_{data['timestamp']}_{data['id']}.jpg"
        # Write the image to the output path
        cv2.imwrite(image_path, data["front_csi_image"])
        # replace the image with the image path
        data["front_csi_image"] = image_path

    def preprocess(self, data: dict, index: int) -> None:
        data["id"] = index
        self._convert_data_format(data)
        self._write_image(data)

    def execute(self, data_queue: MPQueue) -> None:
        if not data_queue.empty():
            data: dict = data_queue.get()
            if data["timestamp"] != self.last_timestamp:
                self.last_timestamp = data["timestamp"]
                self.data_writer.write_data()
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
        qlabs: QuanserInteractiveLabs,
        throttle_coeff: float = 0.3, 
        steering_coeff: float = 0.5
    ) -> None:
        super().__init__(throttle_coeff, steering_coeff)
        # initialize the variables
        self.completed_task_length: int = 0
        self.noise: np.ndarray = np.zeros(2)
        self.action: np.ndarray = np.zeros(2)
        self.observation: dict = {}
        self.data_queue: Queue = Queue(5)
        # initialize the car components
        self.qlabs: QuanserInteractiveLabs = qlabs
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

    def handle_observation(self, orig: np.ndarray, rot: np.ndarray) -> None:
        # in case of less than 200 waypoints
        if self.next_waypoints.shape[0] < MAX_LOOKAHEAD_INDICES:
            slop = MAX_LOOKAHEAD_INDICES - self.next_waypoints.shape[0]
            self.next_waypoints = np.concatenate([self.next_waypoints, self.waypoints[:slop]])
        # add infos to observation
        try:
            self.observation['waypoints'] = np.matmul(self.next_waypoints[:MAX_LOOKAHEAD_INDICES] - orig, rot)
            self.observation['state'] = self.ego_state
            self.observation['noise'] = self.noise
            self.observation['timestamp'] = self.episode_timestamp
            self.observation['task'] = self.task
            self.observation['current_index'] = self.current_waypoint_index - self.completed_task_length
        except Exception as e:
            pass

    def setup(self, waypoints: np.ndarray, init_waypoint_index: int = 0) -> None:
        self.waypoints: np.ndarray = waypoints
        self.ego_state: np.ndarray = self.get_ego_state()
        orig, _, rot = self.cal_vehicle_state(self.ego_state)
        self.current_waypoint_index: int = init_waypoint_index
        self.next_waypoints: np.ndarray = self.waypoints[self.current_waypoint_index:]
        self.episode_timestamp: str = datetime.now().strftime('%Y%m%d%H%M%S%f')
        self.handle_observation(orig, rot)

    def update_state(self) -> None:
        # get the ego state from the qlabs
        self.ego_state = self.get_ego_state() 
        # get the original and rotation matrix
        orig, yaw, rot = self.cal_vehicle_state(self.ego_state)
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
        self.handle_observation(orig, rot)

    def check_new_tasks(self, task_queue: MPQueue) -> None:
        if task_queue is not None and not task_queue.empty():
            task_data: Tuple[List[int], np.ndarray] = task_queue.get()
            self.task: List[int] = task_data[0]
            print(f"Get the new task {task_data[0]}, task length: {len(task_data[1])}") # get the new task
            self.next_waypoints = np.concatenate([self.waypoints[self.current_waypoint_index:], task_data[1]])
            self.completed_task_length = len(self.waypoints)
            self.episode_timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
            self.waypoints = np.concatenate([self.waypoints, task_data[1]]) # update the waypoints
        else: # we will stop the car if there is no new task
            raise ReachGoalException("Reached the goal")
        
    def handle_data_transmit(self, obs_queue: MPQueue = None) -> None:
        # transmit data to the data writer process
        if obs_queue.full():
            obs_queue.get()
        obs_queue.put(self.observation)

    def execute(self, task_queue: MPQueue = None) -> None:
        start_time: float = time.time()
        action, _ = self.policy.execute(obs=self.observation)
        # execute the control
        self.noise = np.array([0.0, get_yaw_noise(action)])
        self.action[0] = action[0] * self.throttle_coeff
        self.action[1] = action[1] * self.steering_coeff
        self.running_gear.read_write_std(
            throttle=self.action[0] + self.noise[0], 
            steering=self.action[1] + self.noise[1]
        )
        print(f"Action: {self.action[1]}, Noise: {self.noise[1]}, Actual: {self.action[1] + self.noise[1]}")
        self.update_state() # update the state
        if self.current_waypoint_index >= len(self.waypoints) - 50:
            self.check_new_tasks(task_queue)
        # calculate the estimated speed
        # linear_speed: float = self.estimate_speed()
        # calculate the sleep time
        execute_time: float = elapsed_time(start_time)
        sleep_time: float = 0.01 - execute_time
        # realtime_message_output(f"Current index {self.current_waypoint_index - self.completed_task_length}")
        time.sleep(max(sleep_time, 0)) # mock delay
