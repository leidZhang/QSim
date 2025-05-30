import sys
sys.path.insert(0, sys.path[0] + "/..")
from collections import defaultdict

from environment import (
    dag_level_order
)

if __name__ == '__main__':
    graph_dict = {0: [2], 1: [2], 2: [3], 3: []}
    # graph_dict = {1: [2, 3], 2: [], 3: []}
    in_degree = defaultdict(int)
    for u in graph_dict:
        for v in graph_dict[u]:
            in_degree[v] += 1

    print(in_degree)
    order = dag_level_order(graph_dict, in_degree)

    print(order)