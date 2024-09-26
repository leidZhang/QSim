import random
from copy import deepcopy
from multiprocessing import Queue
from queue import Queue as MtQueue
from typing import Tuple, List, Union, Dict, Set

import numpy as np

from core.environment import OnlineQLabEnv, AnomalousEpisodeException
from core.environment.simulator import QLabSimulator
from core.environment.detector import EnvQCarRef, is_collided
from core.roadmap.roadmap import ACCRoadMap
from core.templates import BasePolicy, PolicyAdapter
from .agents import *
from .hazard_decision import *

CROSS_ROAD_AREA: Dict[str, float] = {
    "min_x": -0.7, "max_x": 0.9, "min_y": -0.0, "max_y": 2.2
}
THROTTLE_COEEFS: List[float] = [0.09, 0.064, 0.05]
START_POSES: Dict[int, List[float]] = {
    0: [0.0, 2.118, -np.pi / 2],
    1: [0.264, -0.29, np.pi / 2],
    2: [1.206, 1.082, np.pi],
    3: [-1.036, 0.816, 0.0],
}
RESTRICTED_AREAS: List[Dict[str, float]] = [
    None,
    {"max_x": 0.8, "min_x": -0.5, "max_y": 0.2, "min_y": -0.3},
    {"max_x": 1.3, "min_x": 0.5, "max_y": 1.6, "min_y": 0.5},
    {"max_x": 0.05, "min_x": -1.1, "max_y": 1.6, "min_y": 0},
]
ROUTES: Dict[str, List[int]] = {
    0: [[12, 0, 2], [12, 8, 23], [12, 7, 5]],
    1: [[1, 13, 19, 17], [1, 8, 23], [1, 7, 5]],
    2: [[6, 0, 2], [6, 8, 23], [6, 13, 19, 17]],
    3: [[9, 0, 2], [9, 7, 5], [9, 13, 19, 17]],
}


class AnomalousEpisodeDetector:
    def __init__(self) -> None:
        self.last_pose: np.ndarray = np.array([999, 999, 999, 999, 999, 999])
        self.accumulator: int = 0

    def detect(self, pose: np.ndarray, action: np.ndarray) -> bool:
        if action[0] >= 0.045:
            res: int = 1 if np.array_equal(pose, self.last_pose) else -1
            self.accumulator = max(0, self.accumulator + res) # accumulator more than 0
            self.last_pose = deepcopy(pose)

        if self.accumulator >= 10:
            raise AnomalousEpisodeException("Anomalous episode detected")

    def reset_detector(self) -> None:
        self.last_pose = np.array([999, 999, 999, 999, 999, 999])
        self.accumulator = 0


# TODO: Get all the agent states first then check for collision
class CrossRoadEnvironment(OnlineQLabEnv):
    def __init__(
        self,
        simulator: QLabSimulator,
        roadmap: ACCRoadMap,
        dt: float = 0.05,
        privileged: bool = False
    ) -> None:
        super().__init__(simulator, roadmap, dt, privileged)
        self.car_box: EnvQCarRef = EnvQCarRef()
        self.detector: AnomalousEpisodeDetector = AnomalousEpisodeDetector()
        self.__setup_agents()

    def __setup_agents(self) -> None:
        self.agents: List[CarAgent] = []
        self.agents.append(EgoAgent())
        for i in range(1, len(self.simulator.actors["cars"])):
            self.agents.append(HazardAgent(actor_id=i, qlabs=self.simulator.qlabs))
        self.databus: StateDataBus = StateDataBus(
            self.simulator.qlabs, self.agents, self.dt
        )

    def __reset_agent(self, actor_id: int, agent_states: np.ndarray) -> None:
        # get the waypoints and start node for the agent
        random_route_index: int = random.randint(0, 2)
        task: List[int] = ROUTES[actor_id][random_route_index]
        restricted_area: Dict[str, float] = RESTRICTED_AREAS[actor_id]
        if actor_id != 0:
            random_throttle_index: int = random.choice(list(self.used))
            self.used.remove(random_throttle_index)
            throttle_coeff: float = THROTTLE_COEEFS[random_throttle_index]
            # print(f"Actor {actor_id} is using throttle coefficient {throttle_coeff}")
        else:
            throttle_coeff: float = 0.08

        waypoints: np.ndarray = self.roadmap.generate_path(task)
        # put the car actor to the start point
        pose: List[float] = START_POSES[actor_id]
        location: List[float] = [pose[0], pose[1], 0.0]
        orientation: List[float] = [0.0, 0.0, pose[2]]
        # print(f"Actor {actor_id} is at {location} with orientation {orientation}")
        self.simulator.set_car_pos(actor_id, location, orientation)
        # reset the agent's waypoints
        self.agents[actor_id].reset(waypoints, agent_states)
        self.agents[actor_id].set_restricted_area(restricted_area)
        self.agents[actor_id].set_throttle_coeff(throttle_coeff)

    def __render_raster_map(self, raster_info_queue: Queue) -> None:
        ego_state: np.ndarray = self.agents[0].observation["state"]
        hazard_states: List[np.ndarray] = [agent.observation["state"] for agent in self.agents[1:]]
        rendered_waypoint_list: List[np.ndarray] = [self.agents[0].get_task_trajectory()]
        raster_info_queue.put((ego_state, hazard_states, rendered_waypoint_list))

    def __detect_collision(self, ego_state: np.ndarray) -> bool:
        for agent in self.agents[1:]:
            agent_state: np.ndarray = agent.observation["state"]
            if is_collided(ego_state, agent_state, self.car_box):
                print(f"Collision detected between ego agent and agent {agent.actor_id}")
                return True
        return False

    def __detect_anomalous_episode(self) -> bool:
        pose: np.ndarray = self.agents[0].observation["state"]
        action: np.ndarray = self.agents[0].observation["action"]
        self.detector.detect(pose, action)

    def handle_reward(self, agent: CarAgent) -> Tuple[float, bool]:
        state: np.ndarray = agent.observation["state"]

        # When the ego agent collides with any other agent, it gets a penalty and the episode ends
        if self.__detect_collision(state):
            return 0, True
        # When the ego agent reaches the goal, it gets a reward and the episode ends
        if not is_in_area_aabb(state, CROSS_ROAD_AREA):
            return 1, True

        return 0.0, False

    def get_hazard_info(self) -> None:
        hazard_trajs: List[np.ndarray] = [None]
        hazard_progresses: List[int] = [0]
        for agent in self.agents[1:]:
            hazard_trajs.append(agent.observation["global_waypoints"])
            hazard_progresses.append(agent.observation["progress"])
        return hazard_trajs, hazard_progresses

    def reset(self, raster_info_queue: Queue = None) -> Tuple[dict, float, bool, dict]:
        if raster_info_queue is None:
            raise ValueError("Raster info queue cannot be None")

        # reset the car position in the environment
        self.used: Set[float] = {0, 1, 2}
        self.detector.reset_detector()
        _, reward, done, info = super().reset()
        # get the initial state of the agents
        agent_states: List[np.ndarray] = self.databus.reset()
        # reset the agent in the environment
        indices: List[int] = [1, 2, 3]
        self.__reset_agent(0, agent_states)
        for index in indices:
            self.__reset_agent(index, agent_states)
        self.__render_raster_map(raster_info_queue)
        return self.agents[0].observation, reward, done, info

    def step(self, raster_info_queue: Queue = None) -> Tuple[dict, float, bool, dict]:
        self.__detect_anomalous_episode()

        _, reward, done, info = super().step()
        agent_states: List[np.ndarray] = self.databus.step()
        agent_trajs, agent_progresses = self.get_hazard_info()
        # update the agent's state and do the action
        for agent in reversed(self.agents):
            agent.step(agent_states, agent_trajs, agent_progresses)
        self.__render_raster_map(raster_info_queue)
        self.episode_steps += 1

        reward, done = self.handle_reward(self.agents[0])
        done = self.agents[0].observation["done"] or done
        return self.agents[0].observation, reward, done, info

    def stop_all_agents(self) -> None:
        for agent in self.agents:
            agent.halt_car()

    def set_ego_policy(self, policy: Union[BasePolicy, PolicyAdapter]) -> None:
        self.agents[0].set_policy(policy)

    def set_hazard_policy(self, policy: Union[BasePolicy, PolicyAdapter]) -> None:
        for agent in self.agents[1:]:
            agent.set_policy(deepcopy(policy))
