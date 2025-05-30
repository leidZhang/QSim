import random
from typing import List

import numpy as np
from copy import deepcopy

from core.environment import AnomalousEpisodeException
from agents import CarAgent

SPEEDING_PROB: float = 1.0 # 0.5
VIOLATION_PROB_SPEEDING: float = 1.0 # 0.4
VIOLATION_PROB_NORMAL: float = 0.1

class DepartureMonitor:
    def __init__(self) -> None:
        self.activation_pointer: List[int] = None
        self.intersected_node_groups: List[List[int]] = None

    # TODO: Change to DAG-based approach
    def reset(self, agents: List[CarAgent], intersected_node_groups: List[List[int]]) -> None:
        for agent in agents:
            agent.is_hazard = False
            agent.set_speed_lock(False)
            agent.set_init_state(0)

        self.intersected_node_groups = intersected_node_groups
        self.activation_pointer = 0
        for i, group in enumerate(self.intersected_node_groups):
            if i == 0:
                continue
            for id in group:
                agents[id].set_speed_lock(True)

        self.__assign_hazard_agent(agents)
        # for group_id, unlock_pointer in enumerate(self.activation_pointer):
        #     node_group: List[int] = self.intersected_node_groups[group_id]
        #     activate_agent_id: int = node_group[unlock_pointer]
        #     for agent_id in node_group:
        #         # unlock all nodes except the one that triggered the unlock
        #         if agent_id != activate_agent_id:
        #             print("Locking agent", agent_id)
        #             agents[agent_id].set_speed_lock(True) 
        #         else:
        #             print(f"Unlocking agent {agent_id}")

    # TODO: Change to DAG-based approach
    def step(self, agents: List[CarAgent]) -> None:
        if self.activation_pointer == len(self.intersected_node_groups):
            return

        for id in self.intersected_node_groups[self.activation_pointer]:
            current_agent: CarAgent = agents[id]
            if not current_agent.has_passed_crossroads():
                return
            
        if self.activation_pointer + 1 < len(self.intersected_node_groups):
            self.activation_pointer += 1
            for id in self.intersected_node_groups[self.activation_pointer]:
                current_agent: CarAgent = agents[id]
                current_agent.set_speed_lock(False)

        # for group_id, triggered_pointer in enumerate(self.activation_pointer):
        #     agent_id: int = self.intersected_node_groups[group_id][triggered_pointer]
        #     current_agent: CarAgent = agents[agent_id]
        #     # unlock the next agent if the current agent has reached its destination
        #     if current_agent.has_passed_crossroads():
        #         # print(f"Agent {current_agent.id} has passed the crossroads")
        #         self.activation_pointer[group_id] = triggered_pointer - 1 if triggered_pointer >= 0 else 0
        #         next_pointer: int = self.activation_pointer[group_id]
        #         agent_id: int = self.intersected_node_groups[group_id][next_pointer]
        #         next_agent: CarAgent = agents[agent_id]
        #         if next_agent.speed_lock:
        #             print("Unlocking agent", agent_id)
        #             next_agent.set_speed_lock(False)

    def __assign_hazard_agent(self, agents: List[CarAgent]) -> None:
        hazard_results: List[bool] = [False, False, False, False]
        has_speeding_hazard: bool = random.random() < SPEEDING_PROB
        potintial_hazard_agents: List[CarAgent] = []
        for level in self.intersected_node_groups[1:]:
            for id in level:
                if id == 0:
                    continue
                potintial_hazard_agents.append(agents[id])

        if len(potintial_hazard_agents) == 0:
            return

        hazard_agent: CarAgent = random.choice(potintial_hazard_agents)
        if has_speeding_hazard:
            print(f"Agent {hazard_agent.id} is speeding")
            hazard_agent.set_init_state(3)
            is_hazard: bool = random.random() < VIOLATION_PROB_SPEEDING
        else:
            is_hazard: bool = random.random() < VIOLATION_PROB_NORMAL

        hazard_results[hazard_agent.id] = is_hazard
        if is_hazard:
            hazard_agent.is_hazard = True
            hazard_agent.set_speed_lock(False)
            print(f"Agent {hazard_agent.id} is hazardous")
        else:
            hazard_agent.set_speed_lock(True)


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
