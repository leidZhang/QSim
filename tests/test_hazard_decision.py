import numpy as np

from generator.hazard_decision import HazardDetector
from generator.hazard_decision import get_relative_polar_coordinates


def test_get_relative_polar_coordinates_1():
    ego_orig = np.array([0, 0])
    agent_orig = np.array([1, 1])
    cord: np.ndarray = get_relative_polar_coordinates(ego_orig, 0, agent_orig)
    assert cord[0] == np.linalg.norm(ego_orig - agent_orig), "Polar distance should be sqrt(2)"
    assert abs(cord[1]) == np.pi / 4, "Polar angle not correct"


def test_get_relative_polar_coordinates_2():
    ego_orig = np.array([1, 1])
    agent_orig = np.array([0, 0])
    cord: np.ndarray = get_relative_polar_coordinates(ego_orig, 0, agent_orig)
    assert cord[0] == np.linalg.norm(ego_orig - agent_orig), "Polar distance should be sqrt(2)"
    assert abs(cord[1]) == np.pi / 4 * 3, "Polar angle not correct"


def test_get_relative_polar_coordinates_3():
    ego_orig = np.array([0, 0])
    agent_orig = np.array([1, 1])
    cord: np.ndarray = get_relative_polar_coordinates(ego_orig, np.pi, agent_orig)

    assert cord[0] == np.linalg.norm(ego_orig - agent_orig), "Polar distance should be sqrt(2)"
    assert abs(cord[1]) == np.pi / 4 * 3, "Polar angle not correct"


# def test_get_hazard_decision_1():
#     detector: HazardDetector = HazardDetector(0.1)
#     ego_state = np.array([0, 0, 0])
#     agent_states = [np.array([0, 0.05, 0])]

#     hazard_decision = detector.get_hazard_decision(ego_state, agent_states)
#     assert hazard_decision == 3, "Hazard decision should be 0 since there is hazard"


# def test_get_hazard_decision_2():
#     detector: HazardDetector = HazardDetector(0.1)
#     ego_state = np.array([0, 0, 0])
#     agent_states = [np.array([0.5, 0.5, 0])]

#     hazard_decision = detector.get_hazard_decision(ego_state, agent_states)
#     assert hazard_decision == 3, "Hazard decision should be 1 since there is no hazard"


# def test_get_hazard_decision_3():
#     detector: HazardDetector = HazardDetector(0.1)
#     ego_state = np.array([1, 1, np.pi])
#     agent_states = [np.array([0.5, 0.5, 0])]

#     hazard_decision = detector.get_hazard_decision(ego_state, agent_states)
#     assert hazard_decision == 3, "Hazard decision should be 1 since there is no hazard"