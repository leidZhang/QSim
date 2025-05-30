from typing import List, Dict, Tuple, Set
from collections import defaultdict, deque

import numpy as np
from shapely.geometry import Polygon

from core.roadmap.raster_map import to_pixel
from settings import REFERENCE_POSE

from .decision import PriorityNode

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


def build_graph(node_edges: List[tuple], node_list: List[PriorityNode]) -> Tuple[defaultdict, defaultdict]:
    graph_dict: Dict[int, List[int]] = defaultdict(list)
    in_degree: Dict[int, int] = defaultdict(int)
    # for i in range(4):
    #     in_degree[i] += 0
    #     graph_dict[i] = []

    for edge in node_edges:
        u: PriorityNode = node_list[edge[0]]
        v: PriorityNode = node_list[edge[1]]
        if u.has_higher_priority(v):
            graph_dict[u.id].append(v.id)
            in_degree[v.id] += 1
        else:
            graph_dict[v.id].append(u.id)
            in_degree[u.id] += 1

    return graph_dict, in_degree

def dag_level_order(graph: defaultdict, in_degree: defaultdict) -> List[List[int]]:
    queue: deque = deque()
    for node in graph:
        if in_degree[node] == 0:
            queue.append((node, 0)) 

    levels: defaultdict = defaultdict(list)
    while queue:
        node, level = queue.popleft()
        levels[level].append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append((neighbor, level + 1))

    return [levels[lvl] for lvl in sorted(levels.keys())]

def check_corner_order(intersected_trajs: List[List[int]], ordered_groups: List[List[int]]) -> Tuple[bool, Set[int]]:
    intersected_indices = set()
    for pair in intersected_trajs:
        for i in pair:
            intersected_indices.add(i)
    
    for level in ordered_groups:
        for i in level:
            if i not in intersected_indices:
                continue
            intersected_indices.remove(i)

    if len(intersected_indices) == 0:
        return True, intersected_indices
    return False, intersected_indices

def solve_corner_order(unsolved_indices: List[int], ordered_levels: List[List[int]]) -> List[List[int]]:
    num: int = len(unsolved_indices)
    for _ in range(num):
        i: int = unsolved_indices.pop()
        ordered_levels.append([i])
    return ordered_levels

def get_safe_behavior_orders_dag(trajs: List[np.ndarray], nodes: List[PriorityNode]) -> List[list]:
    intersected_trajs: List[int] = []
    for i in range(len(trajs)):
        for j in range(i+1, len(trajs)):
            # print(f"Comparing trajectories {i} and {j}")
            if is_waypoint_intersected(trajs[i][:500], trajs[j][:500]):
                # print(f"Trajectory {i} and {j} intersected")
                intersected_trajs.append([i, j])

    print(f"Intersected trajectories: {intersected_trajs}")

    graph_dict, in_degree = build_graph(intersected_trajs, nodes)
    ordered_levels: List[int] = dag_level_order(graph_dict, in_degree)
    solvable, unsolved_indices = check_corner_order(intersected_trajs, ordered_levels)
    if not solvable:
        print(f"Unsolvable corner order: {unsolved_indices}")
        ordered_levels = solve_corner_order(unsolved_indices, ordered_levels)

    return ordered_levels

def get_safe_behavior_orders_linear(trajs: List[np.ndarray], nodes: List[PriorityNode]) -> List[list]:
    intersected_trajs: List[int] = []
    for i in range(len(trajs)):
        for j in range(i+1, len(trajs)):
            print(f"Comparing trajectories {i} and {j}")
            if is_waypoint_intersected(trajs[i][:500], trajs[j][:500]):
                print(f"Trajectory {i} and {j} intersected")
                intersected_trajs.append([i, j])

    print(f"Intersected trajectories: {intersected_trajs}")

    intersected_groups: List[list] = merge_lists(intersected_trajs)
    sorted_node_groups: List[list] = []
    for group in intersected_groups:
        node_group: List[PriorityNode] = [nodes[i] for i in group]
        node_group = sort_nodes(node_group)
        sorted_node_groups.append(node_group)
    
    return sorted_node_groups


def create_standard_region(center: np.ndarray) -> Polygon:
    orig, yaw = center[:2], center[2] 
    yaw = yaw / 180 * np.pi
    corners: np.ndarray = np.array([
        [-0.15, -0.15],
        [0.15, -0.15],
        [0.15, 0.15],
        [-0.15, 0.15],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    rot_mat: np.ndarray = np.array([
        [c, s],
        [-s, c]
    ])
    corners_rot: np.ndarray = np.matmul(corners, rot_mat) + orig
    return Polygon(corners_rot)

