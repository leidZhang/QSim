import sys
sys.path.insert(0, sys.path[0] + "/..")
from typing import List

from environment import (
    PriorityNode,
    init_nodes,
    sort_nodes
)

def main() -> None:
    node_list: List[PriorityNode] = init_nodes()
    
    node_list[0].set_action("right")
    node_list[1].set_action("straight")
    node_list[2].set_action("straight")
    node_list[3].set_action("straight")
    
    sorted_node_list: List[PriorityNode] = sort_nodes(node_list)
    for node in sorted_node_list:
        print(node)
    
if __name__ == '__main__':
    main()