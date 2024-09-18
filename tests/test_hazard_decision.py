from typing import List

import numpy as np

from core.roadmap import ACCRoadMap
from core.roadmap.raster_map import to_pixel
from generator.hazard_decision import HazardDetector

roadmap: ACCRoadMap = ACCRoadMap()
detector: HazardDetector = HazardDetector()

def get_test_trajs(task_1: List[int], task_2: List[int]) -> None:
    ego_traj: np.ndarray = roadmap.generate_path(task_1)
    hazard_traj: np.ndarray = roadmap.generate_path(task_2)
    return ego_traj, hazard_traj


def is_traj_intersected(ego_traj: np.ndarray, hazard_traj: np.ndarray) -> bool:
    return detector._is_waypoint_intersected(ego_traj, hazard_traj)


def test_is_traj_intersected_1() -> None:
    task_1, task_2 = [12, 0, 2], [1, 8, 10]
    ego_traj, hazard_traj = get_test_trajs(task_1, task_2)
    ego_wp, hazard_wp = ego_traj[:200], hazard_traj[:200]
    res: bool = is_traj_intersected(ego_wp, hazard_wp)
    # print(res)
    assert res == True, "Trajectories are intersected"


def test_is_traj_intersected_2() -> None:
    task_1, task_2 = [12, 0, 2], [1, 13, 19, 17]
    ego_traj, hazard_traj = get_test_trajs(task_1, task_2)
    ego_wp, hazard_wp = ego_traj[:200], hazard_traj[:200]
    res: bool = is_traj_intersected(ego_wp, hazard_wp)
    # print(res)
    assert res == False, "Trajectories are not intersected"


def test_is_traj_intersected_3() -> None:
    task_1, task_2 = [12, 0, 2], [1, 7, 5]
    ego_traj, hazard_traj = get_test_trajs(task_1, task_2)
    ego_wp, hazard_wp = ego_traj[:200], hazard_traj[:200]
    res: bool = is_traj_intersected(ego_wp, hazard_wp)
    # print(res)
    assert res == False, "Trajectories are not intersected"


def test_is_traj_intersected_4() -> None:
    task_1, task_2 = [12, 8, 10], [1, 8, 10]
    ego_traj, hazard_traj = get_test_trajs(task_1, task_2)
    ego_wp, hazard_wp = ego_traj[:200], hazard_traj[:200]
    # print("ego wp", ego_wp[0], ego_wp[-1])
    # print("hazard wp", hazard_wp[0], hazard_wp[-1])
    res: bool = is_traj_intersected(ego_wp, hazard_wp)
    # print(res)
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
    assert res == True, "Trajectories are intersected"


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
    # print("ego wp", ego_wp[0], ego_wp[-1])
    # print("hazard wp", hazard_wp[0], hazard_wp[-1])
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