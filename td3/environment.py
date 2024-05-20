import time
from typing import Tuple

import numpy as np

from qvl.qlabs import QuanserInteractiveLabs

from core.environment import QLabEnvironment
from core.environment.detector import EpisodeMonitor
from .vehicle import WaypointCar
from constants import GOAL_THRESHOLD


class WaypointEnvironment(QLabEnvironment):
    def handle_reward(self, action: list, norm_dist: np.ndarray, ego_state: np.ndarray, dist_ix: int) -> tuple:
        # reset this episode if there's communication issue
        self.detector(action=action, orig=ego_state[:2])
        
        # init params
        done: bool = False
        reward: float = 0.0
        # rewards
        pos = self.vehicle.current_waypoint_index
        region_reward = [1, 4, 2]
        pointer = 0 + (1 if pos > 332 else 0) + (1 if pos > 446 else 0)
        forward_reward = region_reward[pointer] * (pos - self.pre_pos) * 0.125
        reward += forward_reward
        # panelties
        if norm_dist[dist_ix] > 0.05:
            panelty = reward * (norm_dist[dist_ix] / 0.05) * 0.35
            reward -= panelty
        
        # end conditions
        if norm_dist[dist_ix] >= 0.10:
            reward -= 50.0
            done = True
            self.vehicle.halt()  # stop the car # stop the car
        if (np.linalg.norm(self.goal - ego_state[:2]) < GOAL_THRESHOLD and len(self.vehicle.next_waypoints) < 201):
            done = True # stop episode after this step
            self.vehicle.halt()  # stop the car

        return reward, done
    
    # def handle_reward(self, action: list, norm_dist: np.ndarray, ego_state, dist_ix) -> tuple:
    #     # sys.stdout.write(f"\rAction: {action}, Position: {ego_state[:2]}, Start: {self.start_orig}")
    #     # sys.stdout.flush()
    #     self.detector(action=action, orig=ego_state[:2])
    #     done: bool = False
    #     reward: float = 0.0

    #     # Forward reward
    #     if self.prev_dist != np.inf and self.prev_dist - norm_dist[dist_ix] >= -0.01:  # Check if distance to the next waypoint has decreased:

    #         # FORWARD_REWARD V1
    #         pos = self.current_waypoint_index
    #         # print(f'POS: {pos}')
    #         region_reward = [1, 4, 2]
    #         waypoints_range = [(0, 332), (333, 446), (447, 625)]

    #         for i, (start_point, end_point) in enumerate(waypoints_range):
    #             if start_point < pos <= end_point:
    #                 forward_reward = region_reward[i] * (pos - self.pre_pos) * 0.125

    #                 # print(f"FORWARD_REWARD REWARD {forward_reward}")
    #                 reward += forward_reward

    #                 b05_reward = -max(0.0, 4.6 * region_reward[i] * (pos - self.pre_pos) * (norm_dist[dist_ix] + 0.3) ** 4)

    #                 # print(f"0.05 Boundary Reward: {b05_reward}")
    #                 reward += b05_reward

    #                 # print(f'B/F: {"{:.2%}".format(((-b05_reward  / forward_reward)- 0.31) / 0.67)}')

    #         self.pre_pos = pos

    #         # FORWARD_REWARD V2
    #         # forward_reward = action[0] * 24.0
    #         # print(f"FORWARD REWARD {forward_reward}")
    #         # reward += forward_reward


    #         # FORWARD_REWARD V3
    #         # pos = self.current_waypoint_index
    #         #
    #         # rewards_total = [40, 60, 40]
    #         #
    #         # waypoints_range = [(0, 332), (333, 446), (447, 625)]
    #         #
    #         # for i, (start_point, end_point) in enumerate(waypoints_range):
    #         #     if start_point <= pos <= end_point:
    #         #         # print(f'start_point: {start_point}, end_point: {end_point}, pos: {pos}')
    #         #
    #         #         relative_pos = pos - start_point
    #         #         # print(f'relative_pos: {relative_pos}')
    #         #
    #         #         down_term_list = np.arange(0, end_point - start_point) ** 2
    #         #         # print(f'down_term: {down_term}')
    #         #
    #         #         down_term = np.sum(down_term_list)
    #         #         # print(f'down_term: {down_term}')
    #         #
    #         #         up_term = relative_pos ** 2
    #         #         # print(f'up_term: {up_term}')
    #         #
    #         #         r_rate = up_term / down_term
    #         #         # print(f'r_rate: {r_rate}')
    #         #
    #         #         forward_reward = r_rate * rewards_total[i] * 8
    #         #         print(f"FORWARD REWARD {forward_reward}")
    #         #
    #         #         reward += forward_reward
    #         #         break

    #     self.prev_dist = norm_dist[dist_ix]  # Update the previous distance

    #     # Max boundary
    #     if norm_dist[dist_ix] >= 0.10:
    #         max_boundary_reward = -44
    #         # print(f'max_boundary_reward {max_boundary_reward}')
    #         reward += max_boundary_reward
    #         done = True
    #         self.car.read_write_std(0, 0)  # stop the car

    #     # # Boundary reward
    #     # b05_reward = -max(0.0, 4 * (norm_dist[dist_ix] - 0.05))
    #     # reward += b05_reward
    #     # print(f"0.05 Boundary Reward: {b05_reward}")
    #     # b20_reward = -max(0.0, 8 * (norm_dist[dist_ix] - 0.2))
    #     # reward += b20_reward
    #     # print(f"0.20 Boundary Reward: {b20_reward}")

    #     # (no reward) Check if the command is not properly executed by the car
    #     # if abs(action[0]) >= 0.045 and np.array_equal(self.start_orig, ego_state[:2]):
    #     #     raise AnomalousEpisodeException("Anomalous episode detected!")

    #     # (no reward) Reach goal
    #     if (np.linalg.norm(self.goal - ego_state[:2]) < GOAL_THRESHOLD and len(self.next_waypoints) < 201):
    #         done = True  # stop episode after this step
    #         self.car.read_write_std(0, 0)  # stop the car
    #     return reward, done

    def step(self, action: np.ndarray, metrics: dict) -> Tuple[dict, float, bool, dict]:
        episode_done: bool = self.episode_steps >= self.max_episode_steps
        observation, reward, info = self.init_step_params()
        action: np.ndarray = self.vehicle.execute(action)
        time.sleep(0.05)  # sleep for 0.05 seconds

        # privileged information
        if self.privileged:
            ego_state: np.ndarray = self.vehicle.ego_state
            norm_dist: np.ndarray = self.vehicle.norm_dist
            dist_ix: int = self.vehicle.dist_ix
            reward, reward_done = self.handle_reward(action, norm_dist, ego_state, dist_ix)
            episode_done = episode_done or reward_done

        # handle observation
        close_index: int = self.vehicle.current_waypoint_index
        far_index: int = (self.vehicle.current_waypoint_index + 49) % self.waypoint_sequence.shape[0]
        global_close: np.ndarray = self.waypoint_sequence[close_index]
        global_far: np.ndarray = self.waypoint_sequence[far_index]
        observation['state'] = np.concatenate((ego_state, global_close, global_far)) 
        observation["waypoints"] = self.vehicle.observation["waypoints"]
        
        self.episode_steps += 1
        self.pre_pos = self.vehicle.current_waypoint_index
        return observation, reward, episode_done, info
    
    def reset(self) -> Tuple[dict, float, bool, dict]:
        observation, reward, done, info = super().reset()

        # init vehicles, assign proper coeff for throttle and steering if you want
        qlabs: QuanserInteractiveLabs = self.simulator.qlabs
        dt: float = self.simulator.dt
        self.vehicle: WaypointCar = WaypointCar(actor_id=0, dt=dt, qlabs=qlabs, throttle_coeff=0.08)
        self.vehicle.setup(self.waypoint_sequence)
        # init episode params
        self.prev_dist_ix: int = 0
        self.current_waypoint_index: int = 0
        ego_state: np.ndarray = self.vehicle.ego_state
        self.start_orig: np.ndarray = ego_state[:2]
        self.prev_dist = np.inf # set previous distance to infinity
        self.last_orig: np.ndarray = self.start_orig
        self.pre_pos = 0
        # init observations
        global_close: np.ndarray = self.waypoint_sequence[0]
        global_far: np.ndarray = self.waypoint_sequence[49]
        observation['state'] = np.concatenate((ego_state, global_close, global_far))
        observation['waypoints'] = self.vehicle.observation['waypoints'] if self.privileged else None
        # init fault tolerance
        self.detector: EpisodeMonitor = EpisodeMonitor(start_orig=self.start_orig)

        return observation, reward, done, info