from typing import List

import numpy as np
from copy import deepcopy

from core.environment import AnomalousEpisodeException
from .agents import BaseAgent

class DepartureMonitor:
    def __init__(self) -> None:
        self.activation_pointer: List[int] = None
        self.intersected_node_groups: List[List[int]] = None

    def reset(self, agents: List[BaseAgent], intersected_node_groups: List[List[int]]) -> None:
        for agent in agents:
            agent.set_speed_lock(False)

        self.intersected_node_groups = intersected_node_groups
        self.activation_pointer = [len(group) - 1 for group in self.intersected_node_groups]
        for group_id, unlock_pointer in enumerate(self.activation_pointer):
            node_group: List[int] = self.intersected_node_groups[group_id]
            activate_agent_id: int = node_group[unlock_pointer]
            for agent_id in node_group:
                # unlock all nodes except the one that triggered the unlock
                if agent_id != activate_agent_id:
                    print("Locking agent", agent_id)
                    agents[agent_id].set_speed_lock(True) 
                else:
                    print(f"Unlocking agent {agent_id}")

    def step(self, agents: List[BaseAgent]) -> None:
        for group_id, triggered_pointer in enumerate(self.activation_pointer):
            agent_id: int = self.intersected_node_groups[group_id][triggered_pointer]
            current_agent: BaseAgent = agents[agent_id]
            # unlock the next agent if the current agent has reached its destination
            if current_agent.has_passed_area():
                self.activation_pointer[group_id] = triggered_pointer - 1 if triggered_pointer >= 0 else 0
                next_pointer: int = self.activation_pointer[group_id]
                agent_id: int = self.intersected_node_groups[group_id][next_pointer]
                next_agent: BaseAgent = agents[agent_id]
                if next_agent.speed_lock:
                    print("Unlocking agent", agent_id)
                    next_agent.set_speed_lock(False)


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
