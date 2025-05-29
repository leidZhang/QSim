from typing import List, Dict
from collections import defaultdict

import numpy as np

from core.roadmap.raster_map import to_pixel
from settings import REFERENCE_POSE

from .decision_graph import PriorityNode

def init_nodes() -> List[PriorityNode]:
    node_list: List[PriorityNode] = [PriorityNode(i) for i in range(4)]
    for i in range(len(node_list)):
        left_id: int = (i + 1) % len(node_list)
        right_id: int = (i - 1) % len(node_list)
        node_list[i].set_left(node_list[left_id])
        node_list[i].set_right(node_list[right_id])
        print(f"Node {i} has left child {left_id} and right child {right_id}")
    return node_list


def sort_nodes(node_list: List[PriorityNode]) -> List[PriorityNode]:
    n: int = len(node_list)
    for i in range(n):
        for j in range(0, n-1-i):
            if node_list[j].has_higher_priority(node_list[j+1]):
                node_list[j], node_list[j+1] = node_list[j+1], node_list[j]
    return node_list


def is_waypoint_intersected(ego_traj: np.ndarray, hazard_traj: np.ndarray) -> bool:
    ego_polyline: np.ndarray = to_pixel(ego_traj, REFERENCE_POSE[:3], fixed=True)
    hazard_polyline: np.ndarray = to_pixel(hazard_traj, REFERENCE_POSE[:3], fixed=True)

    set1: set = set(map(tuple, ego_polyline))
    set2: set = set(map(tuple, hazard_polyline))
    common_points: set = set1.intersection(set2)
    return len(common_points) != 0


def merge_lists(lists: List[list]) -> List[list]:
    parent: dict = {}

    def find(x: float) -> float:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: float, y: float) -> None:
        parent.setdefault(x, x)
        parent.setdefault(y, y)
        parent[find(x)] = find(y)

    for group in lists:
        for i in range(1, len(group)):
            union(group[0], group[i])

    merged: defaultdict = defaultdict(set)
    for group in lists:
        for item in group:
            root = find(item)
            merged[root].add(item)

    return [sorted(list(group)) for group in merged.values()]


def get_safe_behavior_orders(trajs: List[np.ndarray], nodes: List[PriorityNode]) -> List[list]:
    intersected_trajs: List[int] = []
    for i in range(len(trajs)):
        for j in range(i+1, len(trajs)):
            print(f"Comparing trajectories {i} and {j}")
            if is_waypoint_intersected(trajs[i], trajs[j]):
                print(f"Trajectory {i} and {j} intersected")
                intersected_trajs.append([i, j])

    intersected_groups: List[list] = merge_lists(intersected_trajs)
    sorted_node_groups: List[list] = []
    for group in intersected_groups:
        node_group: List[PriorityNode] = [nodes[i] for i in group]
        node_group = sort_nodes(node_group)
        sorted_node_groups.append(node_group)
    
    return sorted_node_groups


def is_in_area_aabb(state: np.ndarray, area_box: Dict[str, float]) -> bool:
    return area_box["min_x"] <= state[0] <= area_box["max_x"] and area_box["min_y"] <= state[1] <= area_box["max_y"]