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
    """
    The base class for task generator, which is used to generate a task sequence
    based on the given graph.

    Attributes:
    - graph (Dict[int, Set[int]]): The graph to generate the task sequence
    - used (Dict[int, Set[int]]): The used dict to store the used nodes
    - origin (Dict[int, int]): The origin dict to store the origin nodes
    """

    def __init__(self, graph: Dict[int, Set[int]]) -> None:
        """
        Initializes the BaseTaskGenerator object.

        Parameters:
        - graph (Dict[int, Set[int]]): The graph to generate
        """
        self.graph: Dict[int, Set[int]] = deepcopy(graph)
        # generate an empty used dict
        self._prepare_used_dict()
        # generate the origin dict
        self._prepare_origin_dict()

    def _prepare_used_dict(self) -> None:
        """
        Prepare the used dict to store the used nodes.

        Returns:
        - None
        """
        self.used: Dict[int, Set[int]] = {}
        for key, _ in self.graph.items():
            self.used[key] = set()

    def _prepare_origin_dict(self) -> None:
        """
        Prepare the origin dict to store the origin nodes.

        Returns:
        - None
        """
        self.origin: Dict[int, int] = {}
        for key, val in self.graph.items():
            self.origin[key] = len(val)

    def _get_next_node(self, node_sequence: List[int]) -> List[int]:
        """
        Get the next node in the sequence based on the current node sequence.

        Parameters:
        - node_sequence (List[int]): The current node sequence

        Returns:
        - List[int]: The updated node sequence
        """
        final_node_index: int = node_sequence[-1]
        next_index: int = random.choice(tuple(self.graph[final_node_index]))
        self.graph[final_node_index].remove(next_index)
        self.used[final_node_index].add(next_index)
        if len(self.graph[final_node_index]) == 0:
            self.graph[final_node_index] = self.used[final_node_index]
            self.used[final_node_index] = set() # reset the used set
        return node_sequence + [next_index]


class OneTimeTaskGenerator(BaseTaskGenerator):
    """
    The OneTimeTaskGenerator class is used to generate a task sequence based on the
    given graph. It is expected that the task sequence will be generated only once in 
    the iteration.

    Attributes:
    - max_task_length (int): The maximum task length to generate
    - road_map (ACCRoadMap): The roadmap to generate the waypoints
    """

    def __init__(
        self,
        max_task_length: int = 5,
        graph: Dict[int, Set[int]] = ACC_GRAPH_RIGHT,
    ) -> None:
        """
        Initializes the OneTimeTaskGenerator object.

        Parameters:
        - max_task_length (int): The maximum task length to generate
        - graph (Dict[int, Set[int]]): The graph to generate
        """
        super().__init__(graph)
        self.max_task_length: int = max_task_length
        self.road_map: ACCRoadMap = ACCRoadMap()

    def _will_add_node_to_sequence(self, node_sequence: List[int]) -> bool:
        """
        Determine whether to add a new node to the sequence based on the current
        node sequence.

        Parameters:
        - node_sequence (List[int]): The current node sequence

        Returns:
        - bool: Whether to add a new node to the sequence
        """
        if len(node_sequence) == self.max_task_length:
            print(f"New task {node_sequence} generated!")
            return False
        return True

    def get_task(self, start_node_index: int) -> Tuple[List[int], np.ndarray]:
        """
        Get the task sequence and the waypoints based on the start node index.

        Parameters:
        - start_node_index (int): The start node index

        Returns:
        - Tuple[List[int], np.ndarray]: The task sequence and the waypoints
        """
        node_sequence: List[int] = [start_node_index]
        while self._will_add_node_to_sequence(node_sequence):
            node_sequence = self._get_next_node(node_sequence)
        self.graph: Dict[int, Set[int]] = deepcopy(ACC_GRAPH_RIGHT)
        return node_sequence, self.road_map.generate_path(node_sequence)


class TaskDispacher(BaseTaskGenerator):
    """
    The TaskDispacher class is used to dispatch the task sequence based on the
    given graph. It is expected that the task sequence will be dispatched in the
    whole process or thread lifecycle.
    """

    def __init__(
        self,
        start_node: int = 4,
        graph: Dict[int, Set[int]] = ACC_GRAPH_RIGHT,
    ) -> None:
        """
        Initializes the TaskDispacher object.

        Parameters:
        - start_node (int): The start node index
        - graph (Dict[int, Set[int]]): The graph to generate
        """
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
        """
        Determine whether to add a new node to the sequence based on the current
        node sequence. If the length of the node sequence is between 4 and 16 and
        the number of the out edge of the final node is 1, then a new task will 
        be generated.

        Parameters:
        - node_sequence (List[int]): The current node sequence

        Returns:
        - bool: Whether to add a new node to the sequence
        """
        if 4 < len(node_sequence) < 16 and self.origin[node_sequence[-1]] == 1:
            print(f"New task {self.node_sequence} generated!")
            # self.task_length = random.randint(5, 9)
            return False
        return True

    def execute(self, data_queue: Union[Queue, MPQueue]) -> None:
        """
        Execute the task dispatcher to dispatch the task sequence based on the given graph.

        Parameters:
        - data_queue (Union[Queue, MPQueue]): The data queue to transmit the task sequence

        Returns:
        - None
        """
        if self._will_add_node_to_sequence(self.node_sequence):
            self.node_sequence = self._get_next_node(self.node_sequence)
        else: # starting a new task here
            waypoints: np.ndarray = self.road_map.generate_path(self.node_sequence)
            wait_for_empty_queue_space(data_queue) # wait for the queue to have space
            data_queue.put((self.node_sequence, waypoints))
            self.node_sequence = [self.node_sequence[-1]]
