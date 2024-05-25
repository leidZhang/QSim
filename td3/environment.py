import time
from typing import Tuple

import numpy as np

from qvl.qlabs import QuanserInteractiveLabs

from core.environment import QLabEnvironment
from core.environment.detector import EpisodeMonitor
from .vehicle import WaypointCar
from constants import GOAL_THRESHOLD


class WaypointEnvironment(QLabEnvironment):
    # def handle_reward(self, action: list, norm_dist: np.ndarray, ego_state: np.ndarray, dist_ix: int) -> tuple:
    #     # reset this episode if there's communication issue
    #     self.detector(action=action, orig=ego_state[:2])
    #
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
    #
    #     # end conditions
    #     if norm_dist[dist_ix] >= 0.10:
    #         reward -= 50.0
    #         done = True
    #         self.vehicle.halt()  # stop the car # stop the car
    #     if (np.linalg.norm(self.goal - ego_state[:2]) < GOAL_THRESHOLD and len(self.vehicle.next_waypoints) < 201):
    #         done = True # stop episode after this step
    #         self.vehicle.halt()  # stop the car
    #
    #     return reward, done
    
    def handle_reward(self, action: list, norm_dist: np.ndarray, ego_state, dist_ix, global_close, global_far) -> tuple:
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

        # # Max boundary
        if norm_dist[dist_ix] >= 0.10:
            # max_boundary_reward = -44
            # print(f'max_boundary_reward {max_boundary_reward}')
            # reward += max_boundary_reward
            done = True
            self.vehicle.halt()  # stop the car

        # (no reward) Reach goal

        if (np.linalg.norm(self.goal - ego_state[:2]) < GOAL_THRESHOLD and len(self.vehicle.next_waypoints) < 201):
            done = True  # stop episode after this step
            self.vehicle.halt()  # stop the car

        # reward = np.tanh(reward)
        # print(f"reward: {reward}")
        return reward, done



    def step(self, action: np.ndarray, metrics: dict) -> Tuple[dict, float, bool, dict]:
        episode_done: bool = self.episode_steps >= self.max_episode_steps
        observation, reward, info = self.init_step_params()
        action: np.ndarray = self.vehicle.execute(action)
        # time.sleep(0.05)  # sleep for 0.05 seconds

        # extra obs info
        close_index: int = self.vehicle.current_waypoint_index
        far_index: int = (self.vehicle.current_waypoint_index + 39) % self.waypoint_sequence.shape[0]
        global_close: np.ndarray = self.waypoint_sequence[close_index]
        global_far: np.ndarray = self.waypoint_sequence[far_index]
        # print(f'global_far: {global_far}')

        # privileged information
        if self.privileged:
            ego_state: np.ndarray = self.vehicle.ego_state
            norm_dist: np.ndarray = self.vehicle.norm_dist
            dist_ix: int = self.vehicle.dist_ix
            reward, reward_done = self.handle_reward(action, norm_dist, ego_state, dist_ix, global_close, global_far)
            episode_done = episode_done or reward_done

        # handle observation
        observation['state'] = np.concatenate((ego_state, global_close, global_far)) 
        observation["waypoints"] = self.vehicle.observation["waypoints"]
        
        self.episode_steps += 1
        self.pre_pos = self.vehicle.current_waypoint_index
        # print(f'reward: {reward}')
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

    def spawn_on_waypoints(waypoint_sequence: np.ndarray, waypoint_angles: list, actor: QLabsActor,
                           waypoint_num: int = 0, add_deviate: bool = False) -> None:
        """
        Spawns an actor at a specified waypoint with an optional deviation applied.

        This function positions an actor at a given waypoint from a sequence. If 'add_deviate' is True,
        it applies a random deviation to the actor's position and orientation before spawning.

        Parameters:
        - waypoint_sequence (np.ndarray): An array of waypoints, each containing x and y coordinates.
        - waypoint_angles (list): A list of angles corresponding to the orientation at each waypoint.
        - actor (QLabsActor): The actor to be spawned.
        - waypoint_num (int, optional): The index of the waypoint where the actor will be spawned. Defaults to 0.
        - add_deviate (bool, optional): Whether to apply a random deviation to the position and orientation. Defaults to False.

        Raises:
        - Exception: If 'waypoint_sequence' or 'actor' is None.

        Note:
        The function assumes that 'waypoint_sequence' and 'waypoint_angles' are of the same length and that 'waypoint_num' is within
        the valid range of indices for 'waypoint_sequence'.
        """
        if waypoint_sequence is None or actor is None:
            raise Exception("parameters cannot be None")

        x_position: float = waypoint_sequence[waypoint_num][0]
        y_position: float = waypoint_sequence[waypoint_num][1]
        orientation: float = waypoint_angles[waypoint_num]
        ## add random deviate
        # if add_deviate:
        #     deviated_position: list = get_deviate_state([x_position, y_position, orientation])
        #     x_position = deviated_position[0]
        #     y_position = deviated_position[1]
        #     orientation = deviated_position[2]

        # spawn actor on the road map
        actor.spawn_id(
            actorNumber=0,
            location=[x_position, y_position, 0],
            rotation=[0, 0, orientation],
            scale=[.1, .1, .1],
            configuration=0,
            waitForConfirmation=True
        )