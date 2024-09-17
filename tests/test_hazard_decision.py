from typing import List, Callable

import numpy as np

from core.roadmap import ACCRoadMap

roadmap: ACCRoadMap = ACCRoadMap()


def get_test_trajs(task_1: List[int], task_2: List[int]) -> None:
    ego_traj: np.ndarray = roadmap.generate_path(task_1)
    hazard_traj: np.ndarray = roadmap.generate_path(task_2)
    return ego_traj, hazard_traj
    

# TODO: Implement the function is_traj_intersected
def is_traj_intersected(ego_traj: np.ndarray, hazard_traj: np.ndarray) -> bool:
    compare_len: int = min(len(ego_traj), len(hazard_traj))
    waypoint_dists_1: np.ndarray = np.linalg.norm(ego_traj[-compare_len:] - hazard_traj[-compare_len:][::-1], axis=1)
    waypoint_dists_2: np.ndarray = np.linalg.norm(ego_traj[:compare_len] - hazard_traj[:compare_len][::-1], axis=1)
    print(f"min_dist_1", min(waypoint_dists_1), "min_dist_2", min(waypoint_dists_2))
    return np.any(waypoint_dists_1 <= 0.20) or np.any(waypoint_dists_2 <= 0.20)

        
def test_is_traj_intersected_1() -> None:
    task_1, task_2 = [12, 0, 2], [1, 8, 10]
    ego_traj, hazard_traj = get_test_trajs(task_1, task_2)
    ego_wp, hazard_wp = ego_traj[:200], hazard_traj[:200]
    res: bool = is_traj_intersected(ego_wp, hazard_wp)
    print(res)
    assert res == True, "Trajectories are intersected"


def test_is_traj_intersected_2() -> None:
    task_1, task_2 = [12, 0, 2], [1, 13, 19, 17]
    ego_traj, hazard_traj = get_test_trajs(task_1, task_2)
    ego_wp, hazard_wp = ego_traj[:200], hazard_traj[:200]
    res: bool = is_traj_intersected(ego_wp, hazard_wp)
    print(res)
    assert res == False, "Trajectories are not intersected"


def test_is_traj_intersected_3() -> None:
    task_1, task_2 = [12, 0, 2], [1, 7, 5]
    ego_traj, hazard_traj = get_test_trajs(task_1, task_2)
    ego_wp, hazard_wp = ego_traj[:200], hazard_traj[:200]
    res: bool = is_traj_intersected(ego_wp, hazard_wp)
    print(res)
    assert res == False, "Trajectories are not intersected"


def test_is_traj_intersected_4() -> None:
    task_1, task_2 = [12, 8, 10], [1, 8, 10]
    ego_traj, hazard_traj = get_test_trajs(task_1, task_2)
    ego_wp, hazard_wp = ego_traj[:200], hazard_traj[:200]
    print("ego wp", ego_wp[0], ego_wp[-1])
    print("hazard wp", hazard_wp[0], hazard_wp[-1])
    res: bool = is_traj_intersected(ego_wp, hazard_wp)
    print(res)
    assert res == True, "Trajectories are intersected"


def test_is_traj_intersected_5() -> None:
    task_1, task_2 = [12, 8, 10], [1, 13, 19, 17]
    ego_traj, hazard_traj = get_test_trajs(task_1, task_2)
    ego_wp, hazard_wp = ego_traj[:200], hazard_traj[:200]
    # print("ego wp", ego_wp[0], ego_wp[-1])
    # print("hazard wp", hazard_wp[0], hazard_wp[-1])
    res: bool = is_traj_intersected(ego_wp, hazard_wp)
    # print(res)
    assert res == False, "Trajectories are not intersected"


def test_is_traj_intersected_6() -> None:
    task_1, task_2 = [12, 8, 10], [1, 7, 5]
    ego_traj, hazard_traj = get_test_trajs(task_1, task_2)
    ego_wp, hazard_wp = ego_traj[:200], hazard_traj[:200]
    # print("ego wp", ego_wp[0], ego_wp[-1])
    # print("hazard wp", hazard_wp[0], hazard_wp[-1])
    res: bool = is_traj_intersected(ego_wp, hazard_wp)
    # print(res)
    assert res == False, "Trajectories are not intersected"


def test_is_traj_intersected_7() -> None:
    task_1, task_2 = [12, 7, 5], [1, 8, 10]
    ego_traj, hazard_traj = get_test_trajs(task_1, task_2)
    ego_wp, hazard_wp = ego_traj[:200], hazard_traj[:200]
    # print("ego wp", ego_wp[0], ego_wp[-1])
    # print("hazard wp", hazard_wp[0], hazard_wp[-1])
    res: bool = is_traj_intersected(ego_wp, hazard_wp)
    # print(res)
    assert res == False, "Trajectories are not intersected"


def test_is_traj_intersected_8() -> None:
    task_1, task_2 = [12, 7, 5], [1, 13, 19, 17]
    ego_traj, hazard_traj = get_test_trajs(task_1, task_2)
    ego_wp, hazard_wp = ego_traj[:200], hazard_traj[:200]
    # print("ego wp", ego_wp[0], ego_wp[-1])
    # print("hazard wp", hazard_wp[0], hazard_wp[-1])
    res: bool = is_traj_intersected(ego_wp, hazard_wp)
    # print(res)
    assert res == True, "Trajectories are intersected"


def test_is_traj_intersected_9() -> None:
    task_1, task_2 = [12, 7, 5], [1, 7, 5]
    ego_traj, hazard_traj = get_test_trajs(task_1, task_2)
    ego_wp, hazard_wp = ego_traj[:200], hazard_traj[:200]
    # print("ego wp", ego_wp[0], ego_wp[-1])
    # print("hazard wp", hazard_wp[0], hazard_wp[-1])
    res: bool = is_traj_intersected(ego_wp, hazard_wp)
    # print(res)
    assert res == True, "Trajectories are intersected"


def test_is_traj_intersected_10() -> None:
    task_1, task_2 = [12, 0, 2], [9, 7, 5]
    ego_traj, hazard_traj = get_test_trajs(task_1, task_2)
    ego_wp, hazard_wp = ego_traj[:200], hazard_traj[:200]
    # print("ego wp", ego_wp[0], ego_wp[-1])
    # print("hazard wp", hazard_wp[0], hazard_wp[-1])
    res: bool = is_traj_intersected(ego_wp, hazard_wp)
    # print(res)
    assert res == True, "Trajectories are intersected"


def test_is_traj_intersected_11() -> None:
    task_1, task_2 = [12, 0, 2], [9, 13, 19, 17]
    ego_traj, hazard_traj = get_test_trajs(task_1, task_2)
    ego_wp, hazard_wp = ego_traj[:200], hazard_traj[:200]
    # print("ego wp", ego_wp[0], ego_wp[-1])
    # print("hazard wp", hazard_wp[0], hazard_wp[-1])
    res: bool = is_traj_intersected(ego_wp, hazard_wp)
    # print(res)
    assert res == True, "Trajectories are not intersected"


def test_is_traj_intersected_12() -> None:
    task_1, task_2 = [12, 0, 2], [9, 0, 2]
    ego_traj, hazard_traj = get_test_trajs(task_1, task_2)
    ego_wp, hazard_wp = ego_traj[:200], hazard_traj[:200]
    # print("ego wp", ego_wp[0], ego_wp[-1])
    # print("hazard wp", hazard_wp[0], hazard_wp[-1])
    res: bool = is_traj_intersected(ego_wp, hazard_wp)
    # print(res)
    assert res == True, "Trajectories are intersected"


def test_is_traj_intersected_13() -> None:
    task_1, task_2 = [12, 8, 10], [9, 7, 5]
    ego_traj, hazard_traj = get_test_trajs(task_1, task_2)
    ego_wp, hazard_wp = ego_traj[:200], hazard_traj[:200]
    # print("ego wp", ego_wp[0], ego_wp[-1])
    # print("hazard wp", hazard_wp[0], hazard_wp[-1])
    res: bool = is_traj_intersected(ego_wp, hazard_wp)
    # print(res)
    assert res == False, "Trajectories are not intersected"


def test_is_traj_intersected_14() -> None:
    task_1, task_2 = [12, 8, 10], [9, 13, 19, 17]
    ego_traj, hazard_traj = get_test_trajs(task_1, task_2)
    ego_wp, hazard_wp = ego_traj[:200], hazard_traj[:200]
    print("ego wp", ego_wp[0], ego_wp[-1])
    print("hazard wp", hazard_wp[0], hazard_wp[-1])
    res: bool = is_traj_intersected(ego_wp, hazard_wp)
    # print(res)
    assert res == False, "Trajectories are not intersected"


def test_is_traj_intersected_15() -> None:
    task_1, task_2 = [12, 8, 10], [9, 0, 2]
    ego_traj, hazard_traj = get_test_trajs(task_1, task_2)
    ego_wp, hazard_wp = ego_traj[:200], hazard_traj[:200]
    print("ego wp", ego_wp[0], ego_wp[-1])
    print("hazard wp", hazard_wp[0], hazard_wp[-1])
    res: bool = is_traj_intersected(ego_wp, hazard_wp)
    # print(res)
    assert res == False, "Trajectories are not intersected"


def test_is_traj_intersected_16() -> None:
    task_1, task_2 = [12, 7, 5], [9, 7, 5]
    ego_traj, hazard_traj = get_test_trajs(task_1, task_2)
    ego_wp, hazard_wp = ego_traj[:200], hazard_traj[:200]
    # print("ego wp", ego_wp[0], ego_wp[-1])
    # print("hazard wp", hazard_wp[0], hazard_wp[-1])
    res: bool = is_traj_intersected(ego_wp, hazard_wp)
    # print(res)
    assert res == True, "Trajectories are intersected"


def test_is_traj_intersected_17() -> None:
    task_1, task_2 = [12, 7, 5], [9, 13, 19, 17]
    ego_traj, hazard_traj = get_test_trajs(task_1, task_2)
    ego_wp, hazard_wp = ego_traj[:200], hazard_traj[:200]
    # print("ego wp", ego_wp[0], ego_wp[-1])
    # print("hazard wp", hazard_wp[0], hazard_wp[-1])
    res: bool = is_traj_intersected(ego_wp, hazard_wp)
    # print(res)
    assert res == True, "Trajectories are intersected"


def test_is_traj_intersected_18() -> None:
    task_1, task_2 = [12, 7, 5], [9, 0, 2]
    ego_traj, hazard_traj = get_test_trajs(task_1, task_2)
    ego_wp, hazard_wp = ego_traj[:200], hazard_traj[:200]
    # print("ego wp", ego_wp[0], ego_wp[-1])
    # print("hazard wp", hazard_wp[0], hazard_wp[-1])
    res: bool = is_traj_intersected(ego_wp, hazard_wp)
    # print(res)
    assert res == False, "Trajectories are not intersected"