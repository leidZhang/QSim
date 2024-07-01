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

    def handle_reward(self, action: list, norm_dist: np.ndarray, ego_state, dist_ix, global_close, global_far, compare_action) -> tuple:
        # sys.stdout.write(f"\rAction: {action}, Position: {ego_state[:2]}, Start: {self.start_orig}")
        # sys.stdout.flush()
        self.detector(action=action, orig=ego_state[:2])
        done: bool = False
        reward: float = 0.0

        # Forward reward
        # if self.prev_dist != np.inf and self.prev_dist - norm_dist[dist_ix] >= -0.01:  # Check if distance to the next waypoint has decreased:

        # FORWARD_REWARD V1
        pos = self.vehicle.current_waypoint_index
        # print(f'POS: {pos}')
        '''
        region_reward = [1, 4, 2]
        waypoints_range = [(0, 332), (333, 446), (447, 625)]

        for i, (start_point, end_point) in enumerate(waypoints_range):
            # print(f'pos: {pos}, start_point: {start_point}, end_point: {end_point}')
            if start_point < pos <= end_point:
                forward_reward = region_reward[i] * (pos - self.pre_pos) * 0.125

                # print(f"FORWARD_REWARD REWARD {forward_reward}")
                reward += forward_reward

                b05_reward = -max(0.0, 4.6 * region_reward[i] * (pos - self.pre_pos) * (norm_dist[dist_ix] + 0.3) ** 4)

                # print(f"0.05 Boundary Reward: {b05_reward}")
                reward += b05_reward

                # print(f'B/F: {"{:.2%}".format(((-b05_reward  / forward_reward)- 0.31) / 0.67)}')
        '''

        forward_reward = (pos - self.pre_pos) * 0.125
        # print(f"FORWARD_REWARD REWARD {forward_reward}")
        reward += forward_reward

        # compare reward
        # compare_reward = -abs( action[1] - compare_action[1])
        compare_reward = -abs(action[1] - compare_action[1] / 2) * (pos - self.pre_pos) * 0.104
        reward += compare_reward
        # print(f'compare_reward: {compare_reward}')

        # b05_reward = -max(0.0, 1.3 * (pos - self.pre_pos) * (norm_dist[dist_ix] - 0.031))
        # print(f"0.05 Boundary Reward: {b05_reward}")
        # reward += b05_reward

        # print(f'B/F: {"{:.2%}".format(-compare_reward / forward_reward)}')

        self.pre_pos = pos

        self.prev_dist = norm_dist[dist_ix]  # Update the previous distance

        # # cos_v1v2 reward
        # orig, yaw, rot = self.vehicle.cal_vehicle_state(ego_state)
        # # print(f'yaw:{yaw}')
        # # orig: np.ndarray = ego_state[:2]
        # # yaw: float = -ego_state[2]
        # # rot: np.ndarray = np.array([
        # #     [np.cos(yaw), np.sin(yaw)],
        # #     [-np.sin(yaw), np.cos(yaw)]
        # # ])
        #
        # steering_angle = action[1]
        # offset: np.ndarray = np.array([0.0, 0.35])
        # ego_point: np.ndarray = orig + np.matmul(offset, rot)
        # # print(f'global_far {global_far}')
        # # print(f'ego_point {ego_point}')
        # v1: np.ndarray = global_far - ego_point
        # # print(f'v1 {v1}')
        # v2: np.ndarray = np.matmul(np.array([np.sin(steering_angle), np.cos(steering_angle)]), rot) - ego_point
        # # print(f'v2 {v2}')
        #
        # norm_v1: float = np.linalg.norm(v1)
        # # print(f'norm_v1 {norm_v1}')
        # unit_v1: np.ndarray = v1 / norm_v1
        # # print(f'unit_v1 {unit_v1}')
        # norm_v2: float = np.linalg.norm(v2)
        # unit_v2: np.ndarray = v2 / norm_v2
        # cos_v1v2: float = np.dot(unit_v1, unit_v2)
        # # print(f'cos_v1v2 {cos_v1v2}')
        #
        # angle_reward = cos_v1v2
        # # print(f"angle_reward: {angle_reward}")
        #
        # reward += angle_reward


        # Max boundary
        if norm_dist[dist_ix] >= 0.20:
            # max_boundary_reward = -44
            # print(f'max_boundary_reward {max_boundary_reward}')
            # reward += max_boundary_reward
            done = True
            self.vehicle.halt()  # stop the car

        # (no reward) Reach goal
        # end_point =
        if np.linalg.norm(self.goal - ego_state[:2]) < GOAL_THRESHOLD:
            done = True  # stop episode after this step
            self.vehicle.halt()  # stop the car

        # reward = np.tanh(reward)
        # print(f"reward: {reward}")
        return reward, done

    def step(self, action: np.ndarray, metrics: dict, compare_action: np.ndarray) -> Tuple[dict, float, bool, dict]:
        episode_done: bool = self.episode_steps >= self.max_episode_steps
        observation, reward, info = self._init_step_params()
        action: np.ndarray = self.vehicle.execute(action)
        # time.sleep(0.05)  # sleep for 0.05 seconds

        # extra obs info
        close_index: int = self.vehicle.current_waypoint_index  # 0
        far_index: int = (self.vehicle.current_waypoint_index + 49) % self.waypoint_sequence.shape[0]  # 49
        global_close: np.ndarray = self.waypoint_sequence[close_index]
        global_far: np.ndarray = self.waypoint_sequence[far_index]
        # print(f'global_far: {global_far}')

        # privileged information
        if self.privileged:
            ego_state: np.ndarray = self.vehicle.ego_state
            norm_dist: np.ndarray = self.vehicle.norm_dist
            dist_ix: int = self.vehicle.dist_ix
            # print(f'action[1]: {action[1]}')
            # print(f'compare_action[1]: {compare_action[1]}')
            reward, reward_done = self.handle_reward(
                action, norm_dist, ego_state, dist_ix, global_close, global_far, compare_action
            )
            episode_done = episode_done or reward_done

        # handle observation
        state_info: np.ndarray = np.concatenate((ego_state, global_close, global_far))
        observation['state'] = self.recover_state_info(state_info, RECOVER_INDICES)
        observation["waypoints"] = self.vehicle.observation["waypoints"]
        # print(f'close:       {global_close}')
        # print(f'waypoint 1:  {observation["waypoints"][0]}')
        # print(f'far:         {global_far}')
        # print(f'waypoint 50: {observation["waypoints"][49]}')

        self.episode_steps += 1
        self.pre_pos = self.vehicle.current_waypoint_index
        # print(f'reward: {reward}')
        # print(f'observation: {observation}')
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
        global_close: np.ndarray = self.waypoint_sequence[0]
        global_far: np.ndarray = self.waypoint_sequence[49]
        state_info: np.ndarray = np.concatenate((ego_state, global_close, global_far))
        observation['state'] = self.recover_state_info(state_info, RECOVER_INDICES)
        observation['waypoints'] = self.vehicle.observation['waypoints'] if self.privileged else None
        # init fault tolerance
        self.detector: EpisodeMonitor = EpisodeMonitor(start_orig=self.start_orig)
        return observation, reward, done, info
