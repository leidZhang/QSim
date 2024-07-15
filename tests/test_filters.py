import pytest
from typing import List
from core.utils.filters import ThresholdFilter, VariableThresholdFilter


def test_thresh_filter_1() -> None:
    # test use low pass from start
    filter: ThresholdFilter = ThresholdFilter(use_low_pass=True, threshold=0.3)
    filtered_signals: List[float] = []
    input_signals: List[float] = [0, 0.3, 0.4, 0, 0.5, 0.55, 0.03, 0.59, 0.05, 0.63, 0.14, 0.62, 0]
    for signal in input_signals:
        filtered_signals.append(filter(signal))

    # we do not need the first value since it must be 0
    for singals in filtered_signals[1:]:
        assert singals >= 0.3, "Low signals should be filtered!"


def test_variable_threshold_filter_1() -> None:
    # test use low pass from start
    filter: VariableThresholdFilter = VariableThresholdFilter(
        use_low_pass=True, 
        threshold=0.3,
        reduce_factor=0.4
    )
    filtered_signals: List[float] = []
    input_signals: List[float] = [0, 0.3, 0.4, 0, 0.5, 0.55, 0.03, 0.59, 0.05, 0.63, 0.14, 0, 0.62, 0]
    for signal in input_signals:
        filtered_signals.append(filter(signal))

    # we do not need the first value since it must be 0
    print(filtered_signals)
    for singals in filtered_signals[1:]:
        assert singals >= 0.3, "Low signals should be filtered!"
        assert filter.variable_threshold < filter.threshold, "Threshold should be reduced!"