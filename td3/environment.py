import time
from typing import Tuple, List

import numpy as np

from qvl.qlabs import QuanserInteractiveLabs

from core.environment import QLabEnvironment
from core.environment.detector import EpisodeMonitor
from .vehicle import WaypointCar
from constants import GOAL_THRESHOLD, RECOVER_INDICES
import random


class WaypointEnvironment(QLabEnvironment):
    # def handle_reward(self, action: list, norm_dist: np.ndarray, ego_state: np.ndarray, dist_ix: int) -> tuple:
    #     # reset this episode if there's communication issue
    #     self.detector(action=action, orig=ego_state[:2])

    #     # init params
    #     done: bool = False
    #     reward: float = 0.0
    #     # rewards
    #     pos = self.vehicle.current_waypoint_index
    #     region_reward = [1, 4, 2]
    #     pointer = 0 + (1 if pos > 332 else 0) + (1 if pos > 446 else 0)
    #     forward_reward = region_reward[pointer] * (pos - self.pre_pos) * 0.125
    #     reward += forward_reward
    #     # panelties
    #     if norm_dist[dist_ix] > 0.05:
    #         panelty = reward * (norm_dist[dist_ix] / 0.05) * 0.35
    #         reward -= panelty

    #     # end conditions
    #     if norm_dist[dist_ix] >= 0.10:
    #         reward -= 50.0
    #         done = True
    #         self.vehicle.halt()  # stop the car # stop the car
    #     if (np.linalg.norm(self.goal - ego_state[:2]) < GOAL_THRESHOLD and len(self.vehicle.next_waypoints) < 201):
    #         done = True # stop episode after this step
    #         self.vehicle.halt()  # stop the car

    #     return reward, done

    def handle_reward(self, action: list, norm_dist: np.ndarray, ego_state, dist_ix) -> tuple:
        # sys.stdout.write(f"\rAction: {action}, Position: {ego_state[:2]}, Start: {self.start_orig}")
        # sys.stdout.flush()
        self.detector(action=action, orig=ego_state[:2])
        done: bool = False
        reward: float = 0.0

        # FORWARD_REWARD V1
        pos = self.vehicle.current_waypoint_index

        forward_reward = (pos - self.pre_pos) * 0.125
        # print(f"FORWARD_REWARD REWARD {forward_reward}")
        reward += forward_reward

        b05_reward = -max(0.0, 1.3 * (pos - self.pre_pos) * (norm_dist[dist_ix] - 0.031))
        # print(f"0.05 Boundary Reward: {b05_reward}")
        reward += b05_reward

        self.pre_pos = pos
        self.prev_dist = norm_dist[dist_ix]  # Update the previous distance

        # Max boundary
        if norm_dist[dist_ix] >= 0.20:
            # max_boundary_reward = -44
            # print(f'max_boundary_reward {max_boundary_reward}')
            # reward += max_boundary_reward
            done = True
            self.vehicle.halt()  # stop the car

        # (no reward) Reach goal
        if np.linalg.norm(self.goal - ego_state[:2]) < GOAL_THRESHOLD:
            done = True  # stop episode after this step
            self.vehicle.halt()  # stop the car

        return reward, done

    def step(self, action: np.ndarray, metrics: dict) -> Tuple[dict, float, bool, dict]:
        episode_done: bool = self.episode_steps >= self.max_episode_steps
        observation, reward, info = self.init_step_params()
        action: np.ndarray = self.vehicle.execute(action)
        # print(f"Action: {action}")
        # time.sleep(0.05)  # sleep for 0.05 seconds

        # extra obs info
        # close_index: int = self.vehicle.current_waypoint_index  # 0
        # far_index: int = (self.vehicle.current_waypoint_index + 49) % self.waypoint_sequence.shape[0]  # 49
        # global_close: np.ndarray = self.waypoint_sequence[close_index]
        # global_far: np.ndarray = self.waypoint_sequence[far_index]

        # privileged information
        if self.privileged:
            ego_state: np.ndarray = self.vehicle.ego_state
            norm_dist: np.ndarray = self.vehicle.norm_dist
            dist_ix: int = self.vehicle.dist_ix
            reward, reward_done = self.handle_reward(
                action, norm_dist, ego_state, dist_ix
            )
            episode_done = episode_done or reward_done

        # handle observation
        observation['image'] = self.vehicle.observation['image']
        observation['state_info'] = self.vehicle.ego_state
        observation["waypoints"] = self.vehicle.observation["waypoints"]

        self.episode_steps += 1
        self.pre_pos = self.vehicle.current_waypoint_index
        return observation, reward, episode_done, info

    def handle_spawn_pos(self, waypoint_index: int=0) -> Tuple[list, list]:
        # can also call self.spawn_on_node here
        return self.spawn_on_waypoints(waypoint_index)

    def reset(self) -> Tuple[dict, float, bool, dict]:
        # start_index: int = random.randint(420, 750) # change index here
        start_index: int = 0
        # waypoint_index = 420
        # self.goal = self.waypoint_sequence[start_index + 400]
        self.goal = self.waypoint_sequence[-10]
        location, orientation = self.handle_spawn_pos(waypoint_index=start_index)
        observation, reward, done, info = super().reset(location, orientation)

        # init vehicles, assign proper coeff for throttle and steering if you want
        qlabs: QuanserInteractiveLabs = self.simulator.qlabs
        # dt: float = self.simulator.dt
        dt = 0.03
        self.vehicle: WaypointCar = WaypointCar(actor_id=0, dt=dt, qlabs=qlabs, throttle_coeff=0.08)
        self.vehicle.setup(self.waypoint_sequence, start_index)
        # init episode params
        self.prev_dist_ix: int = 0
        ego_state: np.ndarray = self.vehicle.ego_state
        self.start_orig: np.ndarray = ego_state[:2]
        self.prev_dist = np.inf # set previous distance to infinity
        self.last_orig: np.ndarray = self.start_orig
        self.pre_pos: int = self.vehicle.current_waypoint_index
        # init observations
        # global_close: np.ndarray = self.waypoint_sequence[0]
        # global_far: np.ndarray = self.waypoint_sequence[49]
        observation['state_info'] = ego_state
        observation['image'] = self.vehicle.observation['image']
        observation['waypoints'] = self.vehicle.observation['waypoints'] if self.privileged else None
        # init fault tolerance
        self.detector: EpisodeMonitor = EpisodeMonitor(start_orig=self.start_orig)
        return observation, reward, done, info
