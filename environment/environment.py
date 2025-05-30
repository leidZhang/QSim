import time
import random
from copy import deepcopy
from typing import Tuple, List, Union, Dict, Any

from core.environment import OnlineQLabEnv, AnomalousEpisodeException
from core.environment.simulator import QLabSimulator
from core.environment.detector import EnvQCarRef, is_collided
from core.roadmap import ACCRoadMap, get_waypoint_pose
from core.templates import BasePolicy, PolicyAdapter

from agents import CarAgent
from .utils import *
from .message import StateMessageBus
from .monitors import (
    AnomalousEpisodeDetector,
    DepartureMonitor,
    AnomalousEpisodeException,
)

ROUTES: Dict[str, List[int]] = {
    #      straight,     right,         left
    0: [([18, 12, 0, 2, 4], "straight"), ([18, 12, 8, 23, 21], "right"), ([18, 12, 7, 5, 3], "left")],
    1: [([4, 6, 8, 23, 21], "straight"), ([4, 6, 13, 19, 17, 15], "right"), ([4, 6, 0, 2, 4], "left")],
    2: [([3, 1, 13, 19, 17, 15], "straight"), ([3, 1, 7, 5, 3], "right"), ([3, 1, 8, 23, 21], "left")],
    3: [([22, 9, 7, 5, 3], "straight"), ([22, 9, 0, 2, 4], "right"), ([22, 9, 13, 19, 17, 15], "left")],
}
PENDING_AREA_CENTERS: Dict[int, List[float]] = {
    0: [0.0, 2.05, 0],
    1: [1.206, 1.082, 0],    
    2: [0.264, -0.1, 0],
    3: [-0.9, 0.816, 0],
}
DEPARTURE_AREA_CENTERS: Dict[int, List[float]] = {
    4: [0.0, 0.0, 0],
    15: [0.405, 1.75, 0],    
    21: [-0.7, 1.1, 0],
    3: [1.0, 0.816, 0],
}
SPAWN_INDICES: Dict[str, Tuple[int, int]] = {
    0: (50, 20), 
    1: (70, 20), 
    2: (50, 20), 
    3: (180, 150)    
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
        self.has_further_hazard: bool = False
        self.node_list: List[PriorityNode] = init_nodes()
        self.__init_agents()

        # self.start = 0
        # self.start_pending = False
        # self.end_pending = False

    def reset(self) -> Tuple[dict, float, bool, dict]:
        self.start_pending = False
        self.end_pending = False

        for detector in self.detectors:
            detector.reset_detector()        
        observation, reward, done, info = super().reset()  
        agent_states: List[np.ndarray] = self.message_bus.reset()

        tasks, action_types, trajs, departure_area_centers = [], [], [], []
        for i in range(4):
            task, action_type, waypoints, departure_area_center = self.__schedule_agent(i)
            tasks.append(task)
            trajs.append(waypoints)
            departure_area_centers.append(departure_area_center)
            action_types.append(action_type)

        self.__schedule_agent_orders(trajs)
        for i in range(4):
            task: List[int] = tasks
            self.__reset_agent(
                actor_id=i, agent_states=agent_states,
                tasks=task, trajs=trajs
            )

        observation = self.__gather_observations(agent_states, observation)
        observation['violation'] = 1 if self.has_further_hazard else 0

        return observation, reward, done, info
    
    def step(self) -> Tuple[dict, float, bool, dict]:
        self.__detect_anomalous_episode()
        observation, reward, done, info = super().step()
        agent_states: List[np.ndarray] = self.message_bus.step()
        self.departure_monitor.step(self.agents)
        for i, agent in enumerate(self.agents):
            agent.step(agent_states)
        observation = self.__gather_observations(agent_states, observation)
        observation = self.__determine_hazard_status(observation)
        reward, done, info = self.__handle_end_conditions(info)

        # for agent in self.agents:
        #     throttle = agent.observation['action'][0]
        #     if not self.start_pending and throttle == 0:
        #         print("Start pending...")
        #         self.start = time.perf_counter()
        #         self.start_pending = True
        
        # if self.start_pending and not self.end_pending:
        #     for agent in self.agents:
        #         all_zero = all(agent.observation['action'][0] == 0 for agent in self.agents)
        #         if all_zero:
        #             print(f"1. Time elapsed: {time.perf_counter() - self.start}")
        #             print("End pending...")
        #             self.end_pending = True
        #             self.start = time.perf_counter()
        #             break
        # print("Violation prob:", observation['violation'])

        self.episode_steps += 1

        return observation, reward, done, info

    def stop_all_agents(self) -> None:
        for agent in self.agents:
            agent.terminate()
        time.sleep(0.1)

    def set_ego_policy(self, policy: Union[BasePolicy, PolicyAdapter]) -> None:
        self.agents[0].set_policy(policy)

    def set_hazard_policy(self, policy: Union[BasePolicy, PolicyAdapter]) -> None:
        for agent in self.agents[1:]:
            agent.set_policy(deepcopy(policy))

    def __schedule_agent(self, actor_id: int) -> None:
        # get the waypoints and start node for the agent
        random_route_index: int = random.randint(0, 2)
        task: List[int] = ROUTES[actor_id][random_route_index][0]
        action_type: str = ROUTES[actor_id][random_route_index][1]
        self.node_list[actor_id].set_action(action_type)
        departure_area_center: List[float] = DEPARTURE_AREA_CENTERS[task[-1]]
        waypoints: np.ndarray = self.roadmap.generate_path(task)
        print(f"Actor {actor_id} has action type {action_type} going {task[-1]}")
        return task, action_type, waypoints, departure_area_center
    
    def __schedule_agent_orders(self, trajs: List[np.ndarray]) -> List[bool]:
        departure_orders: List[List[int]] = get_safe_behavior_orders_dag(trajs, self.node_list)
        print("The order of departure is:", departure_orders)
        self.departure_monitor.reset(self.agents, departure_orders)
    
    def __determine_hazard_status(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        for agent in self.agents[1:]:
            # print(f"Agent {agent.id} is hazard: {agent.is_hazard} with state {agent.get_state()}")
            if agent.is_hazard and agent.get_state() not in [1, 2]:
                observation['violation'] = 1
                return observation
        observation['violation'] = 0
        return observation

    def __reset_agent(
        self, 
        actor_id: int, 
        agent_states: List[np.ndarray], 
        tasks: List[int],
        trajs: np.ndarray,
    ) -> None:
        agent: CarAgent = self.agents[actor_id]
        # agent.is_hazard = hazard_result[actor_id]
        waypoints, task = trajs[actor_id], tasks[actor_id]
        if agent.get_state() == 3:
            print(f"Agent {actor_id} is spawn further")
            spawn_index: int = SPAWN_INDICES[actor_id][1]
        else:  
            spawn_index: int = SPAWN_INDICES[actor_id][0]
        location, orientation = get_waypoint_pose(waypoints, spawn_index)
        self.simulator.set_car_pos(actor_id, location, orientation)

        departure_area_center: List[float] = DEPARTURE_AREA_CENTERS[task[-1]]
        pending_area_center: List[float] = PENDING_AREA_CENTERS[actor_id]
        agent.reset(waypoints, agent_states, spawn_index)
        departure_area: Polygon = create_standard_region(departure_area_center)
        agent.set_departure_area(departure_area)
        pending_area: Polygon = create_standard_region(pending_area_center)
        agent.set_pending_area(pending_area)

    def __gather_observations(
        self, 
        agent_states: List[np.ndarray], 
        observations: Dict[str, Any]
    ) -> Dict[str, Any]:
        observations['state_info'] = agent_states
        observations['waypoints'] = []
        for agent in self.agents:
            observations['waypoints'].append(agent.observation['waypoints'])
        return observations

    def __init_agents(self) -> None:
        self.agents: List[CarAgent] = []
        for i in range(4):
            self.agents.append(CarAgent(id=i, qlabs=self.simulator.qlabs))
        self.message_bus: StateMessageBus = StateMessageBus(
            self.simulator.qlabs, self.agents, self.dt
        )

    def __handle_end_conditions(self, info: dict) -> Tuple[float, bool]:
        info["collide"] = False
        # When the all agents reache the goal, the episode ends
        done: bool = True
        done = done and self.agents[0].has_passed_crossroads()

        # When the ego agent collides with any other agent, it gets a penalty and the episode ends
        detect_agents: List[CarAgent] = []
        for agent in self.agents:
            if not agent.has_passed_crossroads():
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

    def __detect_anomalous_episode(self) -> bool:
        for i, agent in enumerate(self.agents):
            pose: np.ndarray = agent.observation["state_info"]
            action: np.ndarray = agent.observation["action"]
            self.detectors[i].detect(pose, action)
        # for agent in self.agents[1:]:
        #     if not agent.normal:
        #         raise AnomalousEpisodeException("Anomalous episode detected")
            