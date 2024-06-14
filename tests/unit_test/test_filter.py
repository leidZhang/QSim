import pytest
from typing import List
from core.utils.filters import LowPassFilter


@pytest.mark.unit_test
def test_low_pass_filter_1() -> None:
    filter: LowPassFilter = LowPassFilter(threshold=0.3)
    filtered_signals: List[float] = []
    input_signals: List[float] = [0, 0.3, 0.4, 0, 0.5, 0.55, 0.03, 0.59, 0.05, 0.63, 0.14, 0.62, 0]
    for signal in input_signals:
        filtered_signals.append(filter(signal))

    for singals in filtered_signals[1:]:
        assert singals > 0, "Low signals should be filtered!"