from typing import Tuple

import numpy as np

from qvl.qlabs import QuanserInteractiveLabs

from core.environment import QLabEnvironment
from core.environment.detector import EpisodeMonitor
from constants import GOAL_THRESHOLD, RECOVER_INDICES
from td3.vehicle import WaypointCar


class ReinformerQLabEnv(QLabEnvironment):
    def handle_reward(
        self,
        action: list,
        norm_dist: np.ndarray,
        ego_state: np.ndarray,
        dist_ix: float,
        expert_action: np.ndarray
    ) -> Tuple[float, bool]:
        # detect the anomalous episodes
        self.detector(action=action, orig=ego_state[:2])
        done: bool = False
        reward: float = 0.0

        # calculate the reward based on passed waypoints
        index: int = self.vehicle.current_waypoint_index
        progress: int = index - self.pre_index
        reward += progress * 0.125
        reward += -abs(action[1] - expert_action[1] / 2) * progress * 0.104
        self.pre_index = index

        # Max boundary
        if norm_dist[dist_ix] >= 0.30:
            done = True
            self.vehicle.halt()  # stop the car
        if np.linalg.norm(self.goal - ego_state[:2]) < GOAL_THRESHOLD:
            done = True  # stop episode after this step
            self.vehicle.halt()  # stop the car

        return reward, done

    def step(
        self,
        action: np.ndarray,
        expert_action: np.ndarray
    ) -> Tuple[dict, float, bool, dict]:
        observation, reward, info = self.init_step_params()
        action: np.ndarray = self.vehicle.execute(action)

        ego_state: np.ndarray = self.vehicle.ego_state
        norm_dist: np.ndarray = self.vehicle.norm_dist
        dist_ix: int = self.vehicle.dist_ix
        reward, done = self.handle_reward(
            action, norm_dist, ego_state, dist_ix, expert_action
        )

        ego_state = self.vehicle.ego_state
        local_waypoints = self.vehicle.observation['waypoints']
        observation['state'] = np.concatenate(
            (local_waypoints, ego_state, self.task), axis=1
        )

        return observation, reward, done, info

    def reset(
        self,
        task: list,
        wayponts: np.ndarray,
        start_index: int = 0
    ) -> Tuple[dict, float, bool, dict]:
        # prepare the qcar spawn position
        self.waypoint_sequence: np.ndarray = wayponts
        location, orientation = self.spawn_on_waypoints(waypoint_index=start_index)
        observation, reward, done, info = super().reset(location, orientation)

        # prepare the qcar actor
        qlabs: QuanserInteractiveLabs = self.simulator.qlabs
        self.vehicle: WaypointCar = WaypointCar(
            actor_id=0,
            dt=0.0165,
            qlabs=qlabs,
            throttle_coeff=1,
            steering_coeff=1
        )
        self.vehicle.setup(self.waypoint_sequence, start_index)
        self.pre_index: int = start_index

        # prepare the observations
        self.task: np.ndarray = np.array(task)
        place_holder: np.ndarray = np.full((len(task), 15 - len(task)), -99)
        self.task = np.concatenate(self.task, place_holder)
        ego_state: np.ndarray = self.vehicle.ego_state
        local_waypoints: np.ndarray = self.vehicle.observation['waypoints']
        observation['waypoints'] = local_waypoints
        observation['state'] = np.concatenate(
            (local_waypoints, ego_state, self.task), axis=1
        )

        # init fault tolerance
        self.detector: EpisodeMonitor = EpisodeMonitor(start_orig=location)
        return observation, reward, done, info
