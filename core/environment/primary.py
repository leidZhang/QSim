import time
from typing import Union
from multiprocessing import Queue


from gym import Env
import numpy as np
from pal.products.qcar import QCar

from core.simulator import QLabSimulator
from core.sensor import VirtualCSICamera
from .exception import AnomalousEpisodeException
from constants import MAX_LOOKAHEAD_INDICES, GOAL_THRESHOLD, DEFAULT_MAX_STEPS, MAX_TRAINING_STEPS, max_action, action_v


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QLabEnvironment(Env):
    def __init__(self, dt: float = 0.05, action_size: int = 2, privileged: bool = False) -> None:
        # self.front_csi: VirtualCSICamera = None
        self.action_size: int = action_size
        self.privileged: bool = privileged
        self.max_episode_steps: int = DEFAULT_MAX_STEPS
        self.episode_steps: int = 0
        self.deviate_steps: int = 0
        self.simulator: QLabSimulator = QLabSimulator(dt)

    def setup(self, initial_state: list, sequence: np.ndarray) -> None:
        self.simulator.render_map(initial_state)
        self.set_waypoint_sequence(sequence)

    def set_waypoint_sequence(self, sequence: np.ndarray) -> None:
        self.waypoint_sequence: np.ndarray = sequence
        self.goal: np.ndarray = self.waypoint_sequence[-1]

    def execute_action(self, action: np.ndarray) -> np.ndarray:
        '''
        action type #3: <class 'numpy.ndarray'>
        action shape #3: (2,)
        '''
        action[0] = action_v * (action[0] + 1) / 2 # 0.08 is the max speed of the car
        action[1] = 0.5 * action[1]  # 0.5 is the max steering angle of the car

        self.car.read_write_std(action[0], action[1])
        time.sleep(self.simulator.dt)
        self.last_action_time: float = time.perf_counter()
        return action

    def get_states(self, actor: str) -> tuple:
        ego_state: np.ndarray = self.simulator.get_actor_state(actor_name=actor)
        orig: np.ndarray = ego_state[:2]
        yaw: float = -ego_state[2]
        rot: np.ndarray = np.array([
            [np.cos(yaw), np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)]
        ])
        return orig, yaw, rot

    # def handle_reward(self, action: list, norm_dist: np.ndarray, ego_state, dist_ix) -> tuple:
    #     done: bool = False
    #     reward: float = 0.0
    #
    #     # Forward reward
    #     if self.prev_dist != np.inf:
    #         if self.prev_dist > norm_dist[dist_ix]:  # Check if distance to the next waypoint has decreased
    #             reward += action[0] * 3.0 # (self.prev_dist - norm_dist[dist_ix]) * 100  # Reward for moving closer to the waypoint
    #         elif self.prev_dist == norm_dist[dist_ix]:
    #             reward -= 0.5
    #     self.prev_dist = norm_dist[dist_ix]  # Update the previous distance

    #     # Max boundary
    #     if norm_dist[dist_ix] >= 0.40:
    #         reward -= 40.0
    #         done = True
    #         self.execute_action([0, 0]) # stop the car
    #
    #     if (np.linalg.norm(self.goal - ego_state[:2]) < GOAL_THRESHOLD and len(self.next_waypoints) < 201):
    #         reward += 30.0
    #         done = True # stop episode after this step
    #         self.execute_action([0, 0]) # stop the car
    #
    #     return reward, done

    def handle_reward(self, action: list, norm_dist: np.ndarray, ego_state, dist_ix) -> tuple:
        done: bool = False
        reward: float = 0.0

        # Forward reward
        if self.prev_dist != np.inf and self.prev_dist - norm_dist[dist_ix] >= -0.01:  # Check if distance to the next waypoint has decreased:

            # FORWARD_REWARD V1
            pos = self.current_waypoint_index
            region_reward = [1, 4, 2]
            waypoints_range = [(0, 332), (333, 446), (447, 625)]

            for i, (start_point, end_point) in enumerate(waypoints_range):
                if start_point < pos <= end_point:
                    forward_reward = region_reward[i] * (pos - self.pre_pos) * 0.125

                    print(f"FORWARD_REWARD REWARD {forward_reward}")
                    reward += forward_reward

            self.pre_pos = pos

            # FORWARD_REWARD V2
            # forward_reward = action[0] * 24.0
            # print(f"FORWARD REWARD {forward_reward}")
            # reward += forward_reward


            # FORWARD_REWARD V3
            # pos = self.current_waypoint_index
            #
            # rewards_total = [40, 60, 40]
            #
            # waypoints_range = [(0, 332), (333, 446), (447, 625)]
            #
            # for i, (start_point, end_point) in enumerate(waypoints_range):
            #     if start_point <= pos <= end_point:
            #         # print(f'start_point: {start_point}, end_point: {end_point}, pos: {pos}')
            #
            #         relative_pos = pos - start_point
            #         # print(f'relative_pos: {relative_pos}')
            #
            #         down_term_list = np.arange(0, end_point - start_point) ** 2
            #         # print(f'down_term: {down_term}')
            #
            #         down_term = np.sum(down_term_list)
            #         # print(f'down_term: {down_term}')
            #
            #         up_term = relative_pos ** 2
            #         # print(f'up_term: {up_term}')
            #
            #         r_rate = up_term / down_term
            #         # print(f'r_rate: {r_rate}')
            #
            #         forward_reward = r_rate * rewards_total[i] * 8
            #         print(f"FORWARD REWARD {forward_reward}")
            #
            #         reward += forward_reward
            #         break

        self.prev_dist = norm_dist[dist_ix]  # Update the previous distance

        # Max boundary
        if norm_dist[dist_ix] >= 0.25:
            max_boundary_reward = -80
            print(f'max_boundary_reward {max_boundary_reward}')
            reward += max_boundary_reward
            done = True
            self.car.read_write_std(0, 0)  # stop the car

        # Boundary reward
        b05_reward = -max(0.0, 4 * (norm_dist[dist_ix] - 0.05))
        reward += b05_reward
        print(f"0.05 Boundary Reward: {b05_reward}")
        # b20_reward = -max(0.0, 8 * (norm_dist[dist_ix] - 0.2))
        # reward += b20_reward
        # print(f"0.20 Boundary Reward: {b20_reward}")

        # (no reward) Check if the command is not properly executed by the car
        if abs(action[0]) >= 0.045 and np.array_equal(self.start_orig, ego_state[:2]):
            raise AnomalousEpisodeException("Anomalous episode detected!")

        # (no reward) Reach goal
        if (np.linalg.norm(self.goal - ego_state[:2]) < GOAL_THRESHOLD and len(self.next_waypoints) < 201):
            done = True  # stop episode after this step
            self.car.read_write_std(0, 0)  # stop the car
        return reward, done

    def init_step_params(self) -> tuple:
        observation: dict = {}
        reward: float = 0.0
        info: dict = {}

        return observation, reward, info

    def step(self, action: np.ndarray, metrics: np.ndarray) -> tuple:
        """
        Step the simulation forward

        Parameters:
        - action: np.ndarray: The action to take

        Returns:
        - tuple: The observation, reward, done, and info
        """
        # initialize result variables
        episode_done: bool = self.episode_steps >= self.max_episode_steps
        observation, reward, info = self.init_step_params()
        # execute action
        action = self.execute_action(action) # real qcar action

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
            self.current_waypoint_index = (self.current_waypoint_index + dist_ix) % self.waypoint_sequence.shape[0]
            # self.prev_dist_ix = dist_ix
            self.next_waypoints = self.next_waypoints[dist_ix:]  # clear pasted waypoints

            # add first waypoints if at end
            if self.next_waypoints.shape[0] < MAX_LOOKAHEAD_INDICES:
                # self.goal = self.next_waypoints[-1]
                slop = MAX_LOOKAHEAD_INDICES - self.next_waypoints.shape[0]
                self.next_waypoints = np.concatenate([self.next_waypoints, self.waypoint_sequence[:slop]])

        observation['waypoints'] = np.matmul(self.next_waypoints[:MAX_LOOKAHEAD_INDICES] - orig, rot) if self.privileged else None
        observation['state'] = np.concatenate((ego_state, observation['waypoints'][49])) # TODO: change to min(49, len)
        # print(observation['state'])
        # print(f"Observation: {observation['waypoints']}")
        # observation["image"] = cv2.resize(front_image[:, :, :3], (160, 120))

        # TODO: Extract reward function to a separate method， use the strategy pattern?
        if self.privileged:
            reward, reward_done = self.handle_reward(action, norm_dist, ego_state, dist_ix)

        self.episode_steps += 1
        return observation, reward, reward_done or episode_done, info

    def reset(self) -> tuple:
        """
        Reset the simulation

        Returns:
        - np.ndarray: The observation
        """
        self.simulator.reset_map()
        self.deviate_steps = 0

        self.car: QCar = QCar()
        # reset episode start time
        self.episode_start_time = time.time()
        # initialize result variables
        done: bool = False
        observation, reward, info = self.init_step_params()
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
        self.start_orig = orig
        observation['waypoints'] = np.matmul(self.next_waypoints[:MAX_LOOKAHEAD_INDICES] - orig, rot) if self.privileged else None
        observation['state'] = np.concatenate((ego_state, observation['waypoints'][49]))

        self.prev_dist = np.inf # set previous distance to infinity
        self.last_orig: np.ndarray = orig
        self.last_check_pos: float = time.time()
        self.pre_pos = 0
        return observation, reward, done, info
