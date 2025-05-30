import sys
sys.path.insert(0, sys.path[0] + "/..")
from typing import List, Dict

from environment.decision import (
    PriorityNode,
)
from environment.utils import init_nodes

MOVE_ID, STOP_ID = 0, 1

class MockAgent:
    def __init__(self, id: int) -> None:
        self.id: int = id
        self.speed_lock: bool = False
        self.reached: bool = False

    def step(self) -> None:
        if not self.speed_lock:
            print(f"Agent {self.id} is moving")
            return
        print(f"Agent {self.id} is not moving")

    def lock(self) -> None:
        self.speed_lock = True

    def unlock(self) -> None:
        self.speed_lock = False


class DepartureMonitor:
    def __init__(
        self, 
        trigger_pointers: List[int], 
        intersected_node_groups: List[List[int]], 
        agents: List[MockAgent]
    ) -> None:
        self.trigger_pointers: List[int] = trigger_pointers
        self.intersected_node_groups: List[List[int]] = intersected_node_groups
        self.agents: List[MockAgent] = agents
    
    def step(self, step_id: int) -> None:
        for group_id, triggered_pointer in enumerate(self.trigger_pointers):
            node_id: int = self.intersected_node_groups[group_id][triggered_pointer]
            if step_id >= 10:
                self.trigger_pointers[group_id] = max(0, triggered_pointer - 1)
                next_agent: MockAgent = self.agents[node_id]
                if next_agent.speed_lock:
                    next_agent.unlock()


def init_agents(agents: List[MockAgent], intersected_node_groups: List[List[int]]) -> List[int]:
    trigger_pointers: List[int] = [len(group) - 1 for group in intersected_node_groups]
    for group_id, unlock_pointer in enumerate(trigger_pointers):
        node_group: List[int] = intersected_node_groups[group_id]
        unlock_node: int = node_group[unlock_pointer]
        for node in node_group:
            if node != unlock_node:
                agents[node].lock()
    return trigger_pointers


def main() -> None:
    nodes: List[PriorityNode] = init_nodes()
    nodes[0].set_action("straight")
    nodes[1].set_action("straight")
    nodes[2].set_action("right")
    nodes[3].set_action("right")

    intersected_node_groups: List[List[PriorityNode]] = [
        [3, 0], [2, 1]
    ]
    agents: List[MockAgent] = [MockAgent(i) for i in range(4)]
    trigger_pointers: List[int] = init_agents(agents, intersected_node_groups)
    for agent in agents:
        print(f"Agent {agent.id} is locked: {agent.speed_lock}")

    monitor: DepartureMonitor = DepartureMonitor(
        trigger_pointers, intersected_node_groups, agents
    )
    for i in range(20):
        monitor.step(i)
        print(f"Step {i}")
        for agent in agents:
            agent.step()
        print("====================")


if __name__ == "__main__":
    main()
