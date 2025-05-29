import time
import random
from copy import deepcopy
from typing import Tuple, List, Union, Dict, Set

import numpy as np

from core.environment import OnlineQLabEnv, AnomalousEpisodeException
from core.environment.simulator import QLabSimulator
from core.environment.detector import EnvQCarRef, is_collided
from core.roadmap.roadmap import ACCRoadMap
from core.qcar import QCAR_ACTOR_ID
from core.qcar.virtual import VirtualRuningGear
from core.templates import BasePolicy, PolicyAdapter

from .utils import *
from .decision_graph import PriorityNode
from .agents import BaseAgent, CarAgent
from .databus import StateDataBus
from .monitors import (
    AnomalousEpisodeDetector,
    DepartureMonitor,
    AnomalousEpisodeException,
)

THROTTLE_COEEFS: List[float] = [0.09, 0.064, 0.05]
START_POSES: Dict[int, List[float]] = {
    0: [0.0, 2.118, -np.pi / 2],
    2: [0.264, -0.19, np.pi / 2],
    1: [1.206, 1.082, np.pi],
    3: [-1.036, 0.816, 0.0],
}
DESTINATIONS: Dict[int, Dict[str, float]] = {
    4: {"max_x": 0.27, "min_x": -0.27, "max_y": 0.3, "min_y": 0},
    15: {"max_x": 0.54, "min_x": 0.27, "max_y": 2, "min_y": 1.5},
    21:{"max_x": -0.5, "min_x": -0.9, "max_y": 1.54, "min_y": 1.0},
    3: {"max_x": 1.1, "min_x": 0.9, "max_y": 1.0, "min_y": 0.46},
}
CROSS_ROAD_AREA: Dict[str, float] = {
    "min_x": -1.0, "max_x": 1.0, "min_y": 0.0, "max_y": 2.0
}
ROUTES: Dict[str, List[int]] = {
    #      straight,     right,         left
    0: [([12, 0, 2, 4], "straight"), ([12, 8, 23, 21], "right"), ([12, 7, 5, 3], "left")],
    2: [([1, 13, 19, 17, 15], "straight"), ([1, 7, 5, 3], "right"), ([1, 8, 23, 21], "left")],
    1: [([6, 8, 23, 21], "straight"), ([6, 13, 19, 17, 15], "right"), ([6, 0, 2, 4], "left")],
    3: [([9, 7, 5, 3], "straight"), ([9, 0, 2, 4], "right"), ([9, 13, 19, 17, 15], "left")],
}

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
        self.detectors: List[AnomalousEpisodeDetector] = [AnomalousEpisodeDetector() for _ in range(4)]
        self.departure_monitor: DepartureMonitor = DepartureMonitor()
        self.node_list: List[PriorityNode] = init_nodes()
        self.__setup_agents()

    def reset(self) -> Tuple[dict, float, bool, dict]:
        for detector in self.detectors:
            detector.reset_detector()        
        _, reward, done, info = super().reset()  

        observations: Dict[str, np.ndarray] = {'hazard_states': []}
        agent_states: List[np.ndarray] = self.databus.reset()
        trajs: List[np.ndarray] = []
        for i in range(4):
            self.__reset_agent(i, agent_states)
            trajs.append(self.agents[i].get_waypoints())
        time.sleep(1)

        agent_states: List[np.ndarray] = self.databus.reset()
        intersected_node_groups: List[List[int]] = get_safe_behavior_orders(trajs, self.node_list)
        intersected_node_indices: List[List[int]] = []

        for node_group in intersected_node_groups:
            node_indicies: List[int] = []
            for node in node_group:
                node_indicies.append(node.id)
            intersected_node_indices.append(node_indicies)
        print(f"Intersected groups: {intersected_node_indices}")

        self.departure_monitor.reset(self.agents, intersected_node_indices)

        self.has_outlaw_agent: bool = random.choice([True, False])
        locked_agents: List[CarAgent] = []
        for agent in self.agents[1:]:
            if not agent.speed_lock:
                continue
            locked_agents.append(agent)
        if self.has_outlaw_agent and len(locked_agents) > 0:
            self.outlaw_agent: CarAgent = random.choice(locked_agents)
            self.outlaw_agent.speed_lock = False
            print(f"Outlaw agent {self.outlaw_agent.id} has been released")
        else:
            self.has_outlaw_agent = False
            print(f"No outlaw agent in this episode")
                
        for i in range(4):
            self.agents[i].observation["state_info"] = agent_states[i]
            observations['hazard_states'].append(agent_states[i])

        observations['state_info'] = agent_states[0]
        observations['waypoints'] = self.agents[0].observation['waypoints']

        return observations, reward, done, info

    def step(self) -> Tuple[dict, float, bool, dict]:
        observations: Dict[str, np.ndarray] = {'hazard_states': []}
        self.__detect_anomalous_episode()
        _, reward, done, info = super().step()
        agent_states: List[np.ndarray] = self.databus.step()
        self.departure_monitor.step(self.agents)
        for i, agent in enumerate(self.agents):
            if i == 0:
                observations['waypoints'] = agent.observation['waypoints']
                agent.observation["state_info"] = agent_states[i]
            else:
                observations['hazard_states'].append(agent_states[i])
            agent.step(agent_states)
        self.episode_steps += 1

        reward, done, info = self.handle_end_conditions(info)
        if self.has_outlaw_agent and self.outlaw_agent.has_passed_area():
            print(f"Outlaw agent {self.outlaw_agent.id} has passed the area")
            self.has_outlaw_agent = False

        observations['violation'] = 1 if self.has_outlaw_agent else 0
        observations['state_info'] = agent_states
        print(observations['violation'], self.agents[0].observation["action"])

        return observations, reward, done, info

    def handle_end_conditions(self, info: dict) -> Tuple[float, bool]:
        info["collide"] = False
        # When the all agents reache the goal, the episode ends
        state_info: List[np.ndarray] = []
        done: bool = True
        done = done and self.agents[0].has_passed_area()
        # for i, agent in enumerate(self.agents):
        #     state: np.ndarray = agent.observation["state_info"]
            # in_region: bool = agent.is_in_region(state)
        #     state_info.append(state) if in_region else state_info.append(None)
        #     done = done and not in_region
        # info["state_info"] = state_info

        # When the ego agent collides with any other agent, it gets a penalty and the episode ends
        detect_agents: List[CarAgent] = []
        for agent in self.agents:
            state: np.ndarray = agent.observation["state_info"]
            if is_in_area_aabb(state, CROSS_ROAD_AREA):
                detect_agents.append(agent)

        for i in range(len(detect_agents)):
            for j in range(i + 1, len(detect_agents)):
                ego_state: np.ndarray = detect_agents[i].observation["state_info"]
                hazard_state: np.ndarray = detect_agents[j].observation["state_info"]
                if is_collided(ego_state, hazard_state, self.car_box):
                    print(f"Collision detected between agent {detect_agents[i].id} and agent {detect_agents[j].id}")
                    info["collide"] = True
                    return 0, True, info

        return 0.0, done, info

    def __reset_agent(self, actor_id: int, agent_states: np.ndarray) -> None:
        # get the waypoints and start node for the agent
        random_route_index: int = random.randint(0, 2)
        task: List[int] = ROUTES[actor_id][random_route_index][0]
        action_type: str = ROUTES[actor_id][random_route_index][1]
        destination_area: Dict[str, float] = DESTINATIONS[task[-1]]
        waypoints: np.ndarray = self.roadmap.generate_path(task)

        # put the car actor to the start point
        pose: List[float] = START_POSES[actor_id]
        location: List[float] = [pose[0], pose[1], 0.0]
        orientation: List[float] = [0.0, 0.0, pose[2]]
        print(f"Actor {actor_id} has action type {action_type} going {task[-1]}")
        self.simulator.set_car_pos(actor_id, location, orientation)

        self.node_list[actor_id].set_action(action_type)
        self.agents[actor_id].reset(waypoints, agent_states)
        self.agents[actor_id].set_area(destination_area)

    def __setup_agents(self) -> None:
        self.agents: List[CarAgent] = []
        self.agents.append(CarAgent(id=0, qlabs=self.simulator.qlabs))
        for i in range(1, len(self.simulator.actors["cars"])):
            self.agents.append(CarAgent(id=i, qlabs=self.simulator.qlabs))
            hazard_running_gear: VirtualRuningGear = VirtualRuningGear(QCAR_ACTOR_ID, i)
            self.agents[i].set_running_gear(hazard_running_gear)
        self.databus: StateDataBus = StateDataBus(
            self.simulator.qlabs, self.agents, self.dt
        )

    def __detect_anomalous_episode(self) -> bool:
        for i, agent in enumerate(self.agents):
            pose: np.ndarray = agent.observation["state_info"]
            action: np.ndarray = agent.observation["action"]
            self.detectors[i].detect(pose, action)
        for agent in self.agents[1:]:
            if not agent.normal:
                raise AnomalousEpisodeException("Anomalous episode detected")
            
    def stop_all_agents(self) -> None:
        for agent in self.agents:
            agent.halt_car(0)
        time.sleep(0.1)

    def set_ego_policy(self, policy: Union[BasePolicy, PolicyAdapter]) -> None:
        self.agents[0].set_policy(policy)

    def set_hazard_policy(self, policy: Union[BasePolicy, PolicyAdapter]) -> None:
        for agent in self.agents[1:]:
            agent.set_policy(deepcopy(policy))
