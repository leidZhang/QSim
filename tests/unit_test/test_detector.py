import pytest
import numpy as np
from core.environment.detector import EpisodeMonitor
from core.environment.exception import AnomalousEpisodeException


@pytest.mark.unit_test
def test_episode_monitor_1() -> None:
    # test action[0] > 0.045 and no position change
    test_init_pos: np.ndarray = np.array([1.4, 1.4])
    test_inputs: list = [[np.array([0.073, 0]), np.array([1.4, 1.4])]] * 15
    monitor: EpisodeMonitor = EpisodeMonitor(test_init_pos)
    with pytest.raises(AnomalousEpisodeException) as exc_info:
        for input in test_inputs:
            monitor(input[0], input[1])
    assert "Error happened in the episode!" in str(exc_info.value)


@pytest.mark.unit_test
def test_episode_monitor_2() -> None:
    # test action[0] = 0.045 and no position change
    test_init_pos: np.ndarray = np.array([1.4, 1.4])
    test_inputs: list = [[np.array([0.045, 0]), np.array([1.4, 1.4])]] * 15
    monitor: EpisodeMonitor = EpisodeMonitor(test_init_pos)
    with pytest.raises(AnomalousEpisodeException) as exc_info:
        for input in test_inputs:
            monitor(input[0], input[1])
    assert "Error happened in the episode!" in str(exc_info.value)


@pytest.mark.unit_test
def test_episode_monitor_3() -> None:
    # test action[0] > 0.045 and have position change
    test_init_pos: np.ndarray = np.array([1.4, 1.4])
    test_inputs: list = [[np.array([0.073, 0]), np.array([1.5, 1.4])]] * 15
    monitor: EpisodeMonitor = EpisodeMonitor(test_init_pos)
    for input in test_inputs:
        monitor(input[0], input[1])
    assert monitor.accumulator > 0


@pytest.mark.unit_test
def test_episode_monitor_4() -> None:
    # test action[0] = 0.045 and have position change
    test_init_pos: np.ndarray = np.array([1.4, 1.4])
    test_inputs: list = [[np.array([0.045, 0]), np.array([1.5, 1.4])]] * 15
    monitor: EpisodeMonitor = EpisodeMonitor(test_init_pos)
    for input in test_inputs:
        monitor(input[0], input[1])
    assert monitor.accumulator > 0


@pytest.mark.unit_test
def test_episode_monitor_5() -> None:
    # test action[0] = 0.045 and no position change but haven't full the queue yet
    test_init_pos: np.ndarray = np.array([1.4, 1.4])
    test_inputs: list = [[np.array([0.045, 0]), np.array([1.4, 1.4])]] * 9
    monitor: EpisodeMonitor = EpisodeMonitor(test_init_pos)
    for input in test_inputs:
        monitor(input[0], input[1])
    assert monitor.accumulator == 0


@pytest.mark.unit_test
def test_episode_monitor_6() -> None:
    # test action[0] = 0.045 and have position change but haven't full the queue yet
    test_init_pos: np.ndarray = np.array([1.4, 1.4])
    test_inputs: list = [[np.array([0.045, 0]), np.array([1.5, 1.4])]] * 9
    monitor: EpisodeMonitor = EpisodeMonitor(test_init_pos)
    for input in test_inputs:
        monitor(input[0], input[1])
    assert monitor.accumulator > 0


@pytest.mark.unit_test
def test_episode_monitor_7() -> None:
    # test action[0] > 0.045 and have position change but haven't full the queue yet
    test_init_pos: np.ndarray = np.array([1.4, 1.4])
    test_inputs: list = [[np.array([0.075, 0]), np.array([1.4, 1.4])]] * 9
    monitor: EpisodeMonitor = EpisodeMonitor(test_init_pos)
    for input in test_inputs:
        monitor(input[0], input[1])
    assert monitor.accumulator == 0


@pytest.mark.unit_test
def test_episode_monitor_8() -> None:
    # test action[0] < 0.045 and no position change but haven't full the queue yet
    test_init_pos: np.ndarray = np.array([1.4, 1.4])
    test_inputs: list = [[np.array([0.0, 0]), np.array([1.4, 1.4])]] * 9
    monitor: EpisodeMonitor = EpisodeMonitor(test_init_pos)
    for input in test_inputs:
        monitor(input[0], input[1])
    assert monitor.accumulator > 0


@pytest.mark.unit_test
def test_episode_monitor_9() -> None:
    # test action[0] < 0.045 and no position change
    test_init_pos: np.ndarray = np.array([1.4, 1.4])
    test_inputs: list = [[np.array([0.0, 0]), np.array([1.4, 1.4])]] * 15
    monitor: EpisodeMonitor = EpisodeMonitor(test_init_pos)
    for input in test_inputs:
        monitor(input[0], input[1])
    assert monitor.accumulator > 0


@pytest.mark.unit_test
def test_episode_monitor_10() -> None:
    # test action[0] < 0.045 and no position change, just full the queue
    test_init_pos: np.ndarray = np.array([1.4, 1.4])
    test_inputs: list = [[np.array([0.0, 0]), np.array([1.4, 1.4])]] * 10
    monitor: EpisodeMonitor = EpisodeMonitor(test_init_pos)
    for input in test_inputs:
        monitor(input[0], input[1])
    assert monitor.accumulator > 0


@pytest.mark.unit_test
def test_episode_monitor_11() -> None:
    # test action[0] > 0.045 and no position change, just full the queue
    test_init_pos: np.ndarray = np.array([1.4, 1.4])
    test_inputs: list = [[np.array([0.073, 0]), np.array([1.4, 1.4])]] * 10
    monitor: EpisodeMonitor = EpisodeMonitor(test_init_pos)
    with pytest.raises(AnomalousEpisodeException) as exc_info:
        for input in test_inputs:
            monitor(input[0], input[1])
    assert "Error happened in the episode!" in str(exc_info.value)


@pytest.mark.unit_test
def test_episode_monitor_12() -> None:
    # test action[0] = 0.045 and no position change, just full the queue
    test_init_pos: np.ndarray = np.array([1.4, 1.4])
    test_inputs: list = [[np.array([0.045, 0]), np.array([1.4, 1.4])]] * 10
    monitor: EpisodeMonitor = EpisodeMonitor(test_init_pos)
    with pytest.raises(AnomalousEpisodeException) as exc_info:
        for input in test_inputs:
            monitor(input[0], input[1])
    assert "Error happened in the episode!" in str(exc_info.value)
