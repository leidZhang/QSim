import time
import random
from copy import deepcopy
from queue import Queue
from multiprocessing import Queue as MPQueue
from typing import Dict, List, Set, Union

import numpy as np

from core.roadmap.roadmap import ACCRoadMap
from core.utils.performance import wait_for_empty_queue_space
from .constants import ACC_GRAPH_RIGHT


class TaskDispacher:
    def __init__(
        self,  
        start_node: int = 4,
        graph: Dict[int, Set[int]] = ACC_GRAPH_RIGHT, 
    ) -> None:
        # initialize the roadmap
        self.road_map: ACCRoadMap = ACCRoadMap()
        # initialize the node sequence by appending the start node
        self.node_sequence: List[int] = [start_node]
        # use deepcopy to avoid changing the original dict
        self.graph: Dict[int, Set[int]] = deepcopy(graph)
        # generate an empty used dict
        self._prepare_used_dict() 
        # generate the origin dict
        self._prepare_origin_dict() 
        # generate the random task length
        self.task_length: int = random.randint(3, 9)

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
    
    def _will_add_node_to_sequence(self, node_sequence: List[int]) -> bool:
        if len(node_sequence) == self.task_length:
            print(f"New task {self.node_sequence} generated!")
            self.task_length = random.randint(3, 9)
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

    