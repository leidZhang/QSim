import numpy as np 

from hal.utilities.path_planning import RoadMap

from .constants import X_OFFSET, Y_OFFSET, ACC_SCALE 
from .constants import NODE_POSES_RIGHT_COMMON
from .constants import NODE_POSES_RIGHT_LARGE_MAP 
from .constants import EDGE_CONFIGS_RIGHT_COMMON 
from .constants import EDGE_CONFIGS_RIGHT_LARGE_MAP 


class ACCRoadMap(RoadMap): 
    """
    The road map class for the ACC2024 competition

    Methods:
    - generate_random_cycle: Generates a random cycle from a given starting node
    - generate_path: Generates a path from a given sequence of nodes
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

    def generate_random_cycle(self, start, min_length=3) -> list:
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

        start_node = self.nodes[start]
        cycles = [[start_node] + path for path in dfs(start_node) if len(path) > min_length]
        num_cycles = len(cycles)

        return cycles[np.random.randint(num_cycles)]

    def generate_path(self, sequence: np.ndarray) -> np.array:
        """
        Wraps the generated path as a numpy array object

        Parameters:
        - sequence: np.ndarray: The sequence of nodes

        Returns:
        - np.array: The path as a numpy array
        """
        if type(sequence) == np.ndarray:
            sequence = sequence.tolist()

        return np.array(super().generate_path(sequence)).transpose(1, 0) #[N, (x, y)]