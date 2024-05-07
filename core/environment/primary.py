import time
from typing import Union
from multiprocessing import Queue

from gym import Env
import numpy as np
from pal.products.qcar import QCar

from core.simulator import FullSimulator, PartialSimulator
from core.sensor import VirtualCSICamera
from .constants import MAX_LOOKAHEAD_INDICES, GOAL_THRESHOLD


class QLabEnvironment(Env):
    def __init__(self, dt: float = 0.05, action_size: int = 2, privileged: bool = False) -> None:
        # self.front_csi: VirtualCSICamera = None
        self.action_size: int = action_size
        self.privileged: bool = privileged
        self.max_episode_steps: int = 1000
        self.episode_steps: int = 0
        # self.simulator: QLabSimulator = QLabSimulator(dt)

    def setup(self, initial_state: list, sequence: np.ndarray) -> None:
        self.simulator.render_map(initial_state)
        self.set_waypoint_sequence(sequence)

    def set_waypoint_sequence(self, sequence: np.ndarray) -> None:
        self.waypoint_sequence: np.ndarray = sequence
        self.goal: np.ndarray = self.waypoint_sequence[-1]

    def execute_action(self, action: list) -> None:
        self.car.read_write_std(action[0], action[1])
        time.sleep(self.simulator.dt)
        self.last_action_time: float = time.perf_counter()

    def get_states(self, actor: str) -> tuple:
        ego_state: np.ndarray = self.simulator.get_actor_state(actor_name=actor)
        orig: np.ndarray = ego_state[:2]
        yaw: float = -ego_state[2]
        rot: np.ndarray = np.array([
            [np.cos(yaw), np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)]
        ])
        return orig, yaw, rot

    def step(self, action: Union[np.ndarray, Queue], metrics: np.ndarray) -> tuple:
        """
        Step the simulation forward

        Parameters:
        - action: Union[np.ndarray, Queue]: The action to take

        Returns:
        - tuple: The observation, reward, done, and info
        """
        # initialize result variables
        observation: dict = {}
        done: bool = self.episode_steps >= self.max_episode_steps
        reward: float = 0.0
        info: dict = {}

        # get action from queue if action is a queue
        if type(action) != np.ndarray and not action.empty():
            action: np.ndarray = action.get()
        # execute action and get image
        # if action[0] <= 0.045:
        #     print("low speed!")
        self.execute_action(action)
        # front_image: np.ndarray = self.front_csi.await_image()

        if self.privileged:
            # get ground truth state
            orig, yaw, rot = self.get_states('car')
            waypoints: np.ndarray = np.roll(self.waypoint_sequence, -self.current_waypoint_index, axis=0)[:MAX_LOOKAHEAD_INDICES]
            # get the distance between ego position and waypoints
            norm_dist: np.ndarray = np.linalg.norm(waypoints - orig, axis=1)
            # get the index of the closest waypoint
            dist_ix: int = np.argmin(norm_dist)
            # get state info
            ego_state: np.ndarray = self.simulator.get_actor_state('car')
            reward += (action[0] - 0.045) * 0.5 # temporary reward
            self.current_waypoint_index = (self.current_waypoint_index + dist_ix) % self.waypoint_sequence.shape[0]
            self.prev_dist_ix = dist_ix
            self.next_waypoints = self.next_waypoints[dist_ix:]  # clear pasted waypoints

            # add first waypoints if at end
            if self.next_waypoints.shape[0] < MAX_LOOKAHEAD_INDICES:
                # self.goal = self.next_waypoints[-1]
                slop = MAX_LOOKAHEAD_INDICES - self.next_waypoints.shape[0]
                self.next_waypoints = np.concatenate([self.next_waypoints, self.waypoint_sequence[:slop]])

        observation['state'] = ego_state
        observation['waypoints'] = np.matmul(self.next_waypoints[:MAX_LOOKAHEAD_INDICES] - orig, rot) if self.privileged else None
        # observation["image"] = cv2.resize(front_image[:, :, :3], (160, 120))

        if self.privileged and time.time() - self.last_check_pos >= 0.3:
            reward += action[0] * 2.0 if (self.last_orig != orig).any() else -action[0] * 2.0
            self.last_orig = orig # update position
            self.last_check_pos = time.time() # update timer
        if self.privileged and norm_dist[dist_ix] >= 0.25:
            done = True
            reward -= 30.0 # penalty for not reaching the waypoint
        if self.privileged and (np.linalg.norm(self.goal - ego_state[:2]) < GOAL_THRESHOLD and len(self.next_waypoints) < 201):
            time_taken: float = time.time() - self.episode_start
            if time_taken > 5: # bonus for reach the goal
                reward += 30 - (time_taken - 5) * 0.9
            done = True # stop episode after this step
            self.execute_action([0, 0]) # stop the car

        self.episode_steps += 1
        return observation, reward, done, info

    def reset(self) -> tuple:
        """
        Reset the simulation

        Returns:
        - np.ndarray: The observation
        """
        self.simulator.reset_map()

        self.car: QCar = QCar()
        # self.front_csi: VirtualCSICamera = VirtualCSICamera()
        # front_image: np.ndarray = self.front_csi.await_image()
        # reset episode start time
        self.episode_start_time = time.time()
        # initialize result variables
        observation: dict = {}
        done: bool = self.episode_steps >= self.max_episode_steps
        reward: float = 0.0
        info: dict = {}
        # initialize episode parameters
        self.prev_dist_ix: int = 0
        self.episode_steps = 0
        self.last_action = time.perf_counter()
        self.next_waypoints = self.waypoint_sequence

        self.current_waypoint_index = 0
        self.goal = self.waypoint_sequence[-1]  # x, y coords of goal

        orig, yaw, rot = self.get_states('car')
        ego_state = self.simulator.get_actor_state('car')
        self.episode_start: float = time.time()
        observation['state'] = ego_state
        observation['waypoints'] = np.matmul(self.next_waypoints[:MAX_LOOKAHEAD_INDICES] - orig, rot) if self.privileged else None

        self.prev_dist = np.inf # set previous distance to infinity
        self.last_orig: np.ndarray = orig
        self.last_check_pos: float = time.time()
        return observation, reward, done, info


class GeneratorEnvironment(QLabEnvironment):
    def __init__(self, dt: float = 0.05, action_size: int = 2, privileged: bool = False) -> None:
        super().__init__(dt, action_size, privileged)
        self.simulator: FullSimulator = FullSimulator(dt)


class TrainerEnvironment(QLabEnvironment):
    def __init__(self, dt: float = 0.05, action_size: int = 2, privileged: bool = False) -> None:
        super().__init__(dt, action_size, privileged)
        self.simulator: PartialSimulator = PartialSimulator(dt)
