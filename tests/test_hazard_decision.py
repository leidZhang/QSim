from typing import List, Dict

import cv2
import numpy as np

from core.roadmap import ACCRoadMap
from core.roadmap.raster_map import to_pixel
from generator.env_raster_map import CREnvRasterMap, CROSS_ROARD_RATIAL

CR_REFERENCE_POINT: np.ndarray = np.array([0.15, 0.950, np.pi])
CR_MAP_SIZE: tuple = (384, 384, 3)
CR_MAP_PARAMS: dict = {
    "lanes": ((255, 255, 255), 1),
    "hazards": ((255, 0, 255), 2),
    "waypoints": ((255, 255, 0), 3)
}
roadmap: ACCRoadMap = ACCRoadMap()
renderer: CREnvRasterMap = CREnvRasterMap(roadmap, CR_MAP_SIZE, CR_MAP_PARAMS)


def get_test_trajs(task_1: List[int], task_2: List[int]) -> None:
    ego_traj: np.ndarray = roadmap.generate_path(task_1)
    hazard_traj: np.ndarray = roadmap.generate_path(task_2)
    return ego_traj, hazard_traj


def is_traj_intersected(ego_traj: np.ndarray, hazard_traj: np.ndarray) -> bool:
    ego_polyline = to_pixel(ego_traj, CR_REFERENCE_POINT, offsets=(1.55, 0.75), ratial=CROSS_ROARD_RATIAL)
    hazard_polyline = to_pixel(hazard_traj, CR_REFERENCE_POINT, offsets=(1.55, 0.75), ratial=CROSS_ROARD_RATIAL)
    set1 = set(map(tuple, ego_polyline))
    set2 = set(map(tuple, hazard_polyline))
    common_points = set1.intersection(set2)

    return len(common_points) != 0


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