from typing import Dict, Union, Tuple, List, Set

import numpy as np

from hal.utilities.path_planning import RoadMap, RoadMapNode

from .constants import X_OFFSET, Y_OFFSET, ACC_SCALE
from .constants import NODE_POSES_RIGHT_COMMON
from .constants import NODE_POSES_RIGHT_LARGE_MAP
from .constants import EDGE_CONFIGS_RIGHT_COMMON
from .constants import EDGE_CONFIGS_RIGHT_LARGE_MAP


class ACCRoadMap(RoadMap):
    """
    The road map class for the ACC2024 competition
    """

    def __init__(self) -> None:
        """
        Initializes the ACCRoadMap object
        """
        # parent class initialization
        super().__init__()
        # read nodes and edges
        node_positions: list = NODE_POSES_RIGHT_COMMON + NODE_POSES_RIGHT_LARGE_MAP
        edges: list = EDGE_CONFIGS_RIGHT_COMMON + EDGE_CONFIGS_RIGHT_LARGE_MAP
        # add scaled nodes to acc map
        for position in node_positions:
            position[0] = ACC_SCALE * (position[0] - X_OFFSET)
            position[1] = ACC_SCALE * (Y_OFFSET - position[1])
            self.add_node(position)
        # add scaled edge to acc map
        for edge in edges:
            edge[2] = edge[2] * ACC_SCALE
            self.add_edge(*edge)

    def generate_random_cycle(self, start: int, min_length:int = 3) -> list:
        """
        Generates a random cycle from a given starting node

        Parameters:
        - start: int: The starting node
        - min_length: int: The minimum length of the cycle

        Returns:
        - list: The list of nodes in the cycle
        """
        # depth first search for finding all cycles that start and end at the starting point
        def dfs(start):
            fringe: list = [(start, [])]

            while fringe:
                node, path = fringe.pop()
                if path and node == start:
                    yield path
                    continue
                for next_edges in node.outEdges:
                    next_node = next_edges.toNode
                    if next_node in path:
                        continue
                    fringe.append((next_node, path + [next_node]))

        start_node: RoadMapNode = self.nodes[start]
        cycles: list = [[start_node] + path for path in dfs(start_node) if len(path) > min_length]
        num_cycles: int = len(cycles)
        return cycles[np.random.randint(num_cycles)]

    def generate_path(self, node_sequence: Union[np.ndarray, list]) -> np.array:
        """
        Wraps the generated path as a numpy array object

        Parameters:
        - node_sequence: Union[np.ndarray, list]: The sequence of nodes

        Returns:
        - np.array: The path as a numpy array
        """
        if type(node_sequence) == np.ndarray:
            node_sequence = node_sequence.tolist()

        return np.array(super().generate_path(node_sequence)).transpose(1, 0) #[N, (x, y)]

    def prepare_map_info(self, node_sequence: list) -> Tuple[dict, np.ndarray]:
        """
        Provide the position informations related to the node sequence

        Parameters:
        - node_sequence: Union[np.ndarray, list]: The sequence of nodes

        Returns:
        - Tuple[dict, np.ndarray]: The list of nodes' and waypoints' position
        """
        node_dict: Dict[str, np.ndarray] = {}
        for node_id in node_sequence:
            pose: np.ndarray = self.nodes[node_id].pose
            node_dict[node_id] = pose # x, y, angle

        waypoint_sequence = self.generate_path(node_sequence)
        return node_dict, waypoint_sequence
    