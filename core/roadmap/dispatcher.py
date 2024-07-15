import random
from copy import deepcopy
from queue import Queue
from abc import ABC
from multiprocessing import Queue as MPQueue
from typing import Dict, List, Set, Union, Tuple

import numpy as np

from core.roadmap.roadmap import ACCRoadMap
from core.utils.performance import wait_for_empty_queue_space
from .constants import ACC_GRAPH_RIGHT


class BaseTaskGenerator(ABC):
    def __init__(self, graph: Dict[int, Set[int]]) -> None:
        self.graph: Dict[int, Set[int]] = deepcopy(graph)
        # generate an empty used dict
        self._prepare_used_dict()
        # generate the origin dict
        self._prepare_origin_dict()

    def _prepare_used_dict(self) -> None:
        self.used: Dict[int, Set[int]] = {}
        for key, _ in self.graph.items():
            self.used[key] = set()

    def _prepare_origin_dict(self) -> None:
        self.origin: Dict[int, int] = {}
        for key, val in self.graph.items():
            self.origin[key] = len(val)

    def _get_next_node(self, node_sequence: List[int]) -> List[int]:
        final_node_index: int = node_sequence[-1]
        next_index: int = random.choice(tuple(self.graph[final_node_index]))
        self.graph[final_node_index].remove(next_index)
        self.used[final_node_index].add(next_index)
        if len(self.graph[final_node_index]) == 0:
            self.graph[final_node_index] = self.used[final_node_index]
            self.used[final_node_index] = set() # reset the used set
        return node_sequence + [next_index]


class OneTimeTaskGenerator(BaseTaskGenerator):
    def __init__(
        self,
        max_task_length: int = 5,
        graph: Dict[int, Set[int]] = ACC_GRAPH_RIGHT,
    ) -> None:
        super().__init__(graph)
        self.max_task_length: int = max_task_length
        self.road_map: ACCRoadMap = ACCRoadMap()

    def _will_add_node_to_sequence(self, node_sequence: List[int]) -> bool:
        if len(node_sequence) == self.max_task_length:
            print(f"New task {node_sequence} generated!")
            return False
        return True

    def get_task(self, start_node_index: int) -> Tuple[List[int], np.ndarray]:
        node_sequence: List[int] = [start_node_index]
        while self._will_add_node_to_sequence(node_sequence):
            node_sequence = self._get_next_node(node_sequence)
        self.graph: Dict[int, Set[int]] = deepcopy(ACC_GRAPH_RIGHT)
        return node_sequence, self.road_map.generate_path(node_sequence)


class TaskDispacher(BaseTaskGenerator):
    def __init__(
        self,
        start_node: int = 4,
        graph: Dict[int, Set[int]] = ACC_GRAPH_RIGHT,
    ) -> None:
        super().__init__(graph)
        # initialize the roadmap
        self.road_map: ACCRoadMap = ACCRoadMap()
        # initialize the node sequence by appending the start node
        self.node_sequence: List[int] = [start_node]
        # use deepcopy to avoid changing the original dict
        self.graph: Dict[int, Set[int]] = deepcopy(graph)
        # generate the random task length
        # self.task_length: int = random.randint(5, 9)

    def _will_add_node_to_sequence(self, node_sequence: List[int]) -> bool:
        if 4 < len(node_sequence) < 16 and self.origin[node_sequence[-1]] == 1:
            print(f"New task {self.node_sequence} generated!")
            # self.task_length = random.randint(5, 9)
            return False
        return True

    def execute(self, data_queue: Union[Queue, MPQueue]) -> None:
        if self._will_add_node_to_sequence(self.node_sequence):
            self.node_sequence = self._get_next_node(self.node_sequence)
        else: # starting a new task here
            waypoints: np.ndarray = self.road_map.generate_path(self.node_sequence)
            wait_for_empty_queue_space(data_queue) # wait for the queue to have space
            data_queue.put((self.node_sequence, waypoints))
            self.node_sequence = [self.node_sequence[-1]]
