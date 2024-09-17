import time
import random
from copy import deepcopy
from typing import Tuple, List, Union, Dict, Set

import numpy as np

from core.environment import OnlineQLabEnv
from core.environment.simulator import QLabSimulator
from core.environment.detector import EnvQCarRef, is_collided
from core.roadmap.roadmap import ACCRoadMap
from core.templates import BasePolicy, PolicyAdapter
from .hazard_decision import *
from .env_raster_map import *
from .agents import *

THROTTLE_COEEFS: List[float] = [0.09, 0.064, 0.05]
START_POSES: Dict[int, List[float]] = {
    0: [0.0, 2.118, -np.pi / 2],
    1: [0.264, -0.29, np.pi / 2],
    2: [1.206, 1.082, np.pi],
    3: [-1.036, 0.816, 0.0],
}
ROUTES: Dict[str, List[int]] = {
    0: [[12, 0, 2], [12, 8, 10], [12, 7, 5]],
    1: [[1, 13, 19, 17], [1, 8, 10], [1, 7, 5]],
    2: [[6, 0, 2], [6, 8, 10], [6, 13, 19, 17]],
    3: [[9, 0, 2], [9, 7, 5], [9, 13, 19, 17]],
}


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
        self.renderer: CREnvRasterMap = CREnvRasterMap(self.roadmap, CR_MAP_SIZE, CR_MAP_PARAMS)
        self.detector: HazardDetector = HazardDetector()
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
        if actor_id != 0:
            random_throttle_index: int = random.choice(list(self.used))
            self.used.remove(random_throttle_index)
            throttle_coeff: float = THROTTLE_COEEFS[random_throttle_index]
            print(f"Actor {actor_id} is using throttle coefficient {throttle_coeff}")
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
        self.agents[actor_id].set_throttle_coeff(throttle_coeff)

    def __render_raster_map(self) -> None:
        agent_states: List[np.ndarray] = []
        waypoint_list: List[np.ndarray] = []
        for agent in self.agents[1:]:
            agent_states.append(agent.observation["state"])
            waypoint_list.append(agent.observation["global_waypoints"])
        raster_map, _, _ = self.renderer.draw_map(
            [0.15, 0.950, np.pi, 0.0, 0.0, 0.0], agent_states, waypoint_list
        )
        cv2.imshow("Raster Map", raster_map)
        cv2.waitKey(1)

    def reset(self) -> Tuple[dict, float, bool, dict]:
        # reset the car position in the environment
        self.used: Set[float] = {0, 1, 2}
        _, reward, done, info = super().reset()
        # get the initial state of the agents
        agent_states: List[np.ndarray] = self.databus.reset()
        # reset the agent in the environment
        indices: List[int] = [1, 2, 3]
        self.__reset_agent(0, agent_states)
        for index in indices:
            self.__reset_agent(index, agent_states)
        return self.agents[0].observation, reward, done, info

    def step(self) -> Tuple[dict, float, bool, dict]:
        print("====================================")
        _, reward, done, info = super().step()
        agent_states: List[np.ndarray] = self.databus.step()

        # update the agent's state
        for agent in reversed(self.agents):
            agent.step(agent_states)
        self.episode_steps += 1

        self.__render_raster_map()

        # check for hazard
        hazard_agents: List[CarAgent] = self.agents[1:]
        i: int = 1
        while i < len(hazard_agents):
            j: int = 0
            while j < len(hazard_agents):
                if i == j:
                    j += 1
                    continue
                subject: CarAgent = hazard_agents[i]
                object: CarAgent = hazard_agents[j]
                subject_coeff: int = self.detector.evalueate(subject, object)
                subject.observation["hazard_coeff"] = subject_coeff
                object_coeff: int = self.detector.evalueate(object, subject)
                object.observation["hazard_coeff"] = object_coeff
                j += 1
            i += 1

        # ego_agent_state: np.ndarray = self.agents[0].observation["state"]
        # for agent in self.agents[1:]:
        #     agent_state: np.ndarray = agent.observation["state"]
        #     if is_collided(ego_agent_state, agent_state, self.car_box):
        #         print(f"Collision detected between ego agent and agent {agent.actor_id}")
        #         done = True
        #         break

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