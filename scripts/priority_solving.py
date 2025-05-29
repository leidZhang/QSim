import sys
sys.path.insert(0, sys.path[0] + "/..")
from typing import List, Tuple, Set
from collections import defaultdict

import numpy as np

from core.roadmap import ACCRoadMap
from environment import (
    PriorityNode,
    init_nodes,
    get_safe_behavior_orders
)


def main() -> None:
    roadmap: ACCRoadMap = ACCRoadMap()
    tasks: List[List[int]] = [
        # [12, 8, 23, 21],
        # [1, 7, 5, 3],
        # [6, 13, 19, 17],
        # [9, 7, 5, 3],
        [12, 0, 2, 4],
        [1, 13, 19, 17, 15],
        [6, 13, 19, 17, 15],
        [9, 0, 2, 4]
    ]

    trajs: List[np.ndarray] = [roadmap.generate_path(task) for task in tasks]
    nodes: List[PriorityNode] = init_nodes()
    nodes[0].set_action("straight")
    nodes[1].set_action("straight")
    nodes[2].set_action("right")
    nodes[3].set_action("right")

    intersected_nodes = get_safe_behavior_orders(trajs, nodes)
    for nodes in intersected_nodes:
        print("Pair:")
        for node in nodes:
            print(node)


if __name__ == "__main__":  
    main()