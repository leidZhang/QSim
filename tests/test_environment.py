import math
from typing import Tuple, Dict

import pytest
import numpy as np
from core.environment import QLabEnvironment


def prepare_test_environment() -> QLabEnvironment:
    # test instance
    test_env: QLabEnvironment = QLabEnvironment()
    nodes: Dict[str, np.ndarray] = {
        0: np.array([0, 0, math.pi]),
        1: np.array([-1, -3, math.pi * 7 / 4]),
    }
    waypoints: np.ndarray = np.array([
        [0, 0], [-1, 0], [-2, -1],
        [-2, -2], [-1, -3], [0, -3],
        [1, -2], [1, -1], [0, 0]
    ])
    test_env.set_nodes(nodes)
    test_env.set_waypoint_sequence(waypoints)
    return test_env


def test_cal_waypoint_angle() -> None:
    test_env: QLabEnvironment = prepare_test_environment()
    waypoints: np.ndarray = test_env.waypoint_sequence
    # expectation array
    expect: list = [
        math.pi, math.pi * 5 / 4, math.pi * 3 / 2,
        math.pi * 7 / 4, 0, math.pi / 4,
        math.pi / 2, math.pi * 3 / 4
    ]

    for i in range(len(waypoints) - 1):
        delta_x: float = waypoints[i+1][0] - waypoints[i][0]
        delta_y: float = waypoints[i+1][1] - waypoints[i][1]
        res: float = test_env.cal_waypoint_angle(delta_x, delta_y)
        if res >= 0:
            assert res == pytest.approx(expect[i]), \
                f"Expect: {expect[i]} at index {i}, but got {res}"
        else:
            assert res == pytest.approx(expect[i] - 2 * math.pi), \
                f"Expect: {expect[i] - 2 * math.pi} at index {i}, but got {res}"


def test_spawn_on_waypoints_1() -> None:
    # test the index < 0
    test_env: QLabEnvironment = prepare_test_environment()
    with pytest.raises(ValueError) as exc_info:
        test_env.spawn_on_waypoints(-1)
    assert 'Invalid Waypoint index format' in str(exc_info.value)


def test_spawn_on_waypoints_2() -> None:
    # test the index > len(waypoint)
    test_env: QLabEnvironment = prepare_test_environment()
    with pytest.raises(ValueError) as exc_info:
        test_env.spawn_on_waypoints(10000)
    assert 'Invalid Waypoint index format' in str(exc_info.value)


def test_spawn_waypoints_3() -> None:
    # test conditions with currenct index
    test_env: QLabEnvironment = prepare_test_environment()
    expect: list = [
        ([0, 0, 0], [0, 0, math.pi]),
        ([-1, 0, 0], [0, 0, math.pi * 5 / 4]),
        ([-2, -1, 0], [0, 0, math.pi * 3 / 2]),
        ([-2, -2, 0], [0, 0, math.pi * 7 / 4]),
        ([-1, -3, 0], [0, 0, 0]),
        ([0, -3, 0], [0, 0, math.pi / 4]),
        ([1, -2, 0], [0, 0, math.pi / 2]),
        ([1, -1, 0], [0, 0, math.pi * 3 / 4]),
        ([0, 0, 0], [0, 0, math.pi * 3 / 4]),
    ]
    for i in range(len(test_env.waypoint_sequence) - 1):
        res: Tuple[list, list] = test_env.spawn_on_waypoints(i)
        if res[1][2] < 0: # sometimes value is different but the orientation is the same
            expect[i][1][2] -= 2 * math.pi
        assert res == expect[i], f"Expect: {expect[i]} at index {i}, but got {res}"


def test_spawn_on_nodes_1() -> None:
    # test index not exist
    test_env: QLabEnvironment = prepare_test_environment()
    with pytest.raises(ValueError) as exc_info:
        test_env.spawn_on_nodes(100000)
    assert 'Index does not exist!' in str(exc_info.value)


def test_spawn_on_nodes_2() -> None:
    # test index exist
    test_env: QLabEnvironment = prepare_test_environment()
    expect: Tuple[list, list] = ([0, 0, 0], [0, 0, math.pi])
    res: Tuple[list, list] = test_env.spawn_on_nodes(0)
    assert res == expect, f"Expect: {expect}, but got {res}"
