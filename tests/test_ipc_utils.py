from typing import List
from core.utils.ipc_utils import EventDoubleBuffer


# --------- Test Cases for EventDoubleBuffer ---------
def test_put_1() -> None:
    # test the case when buffer is not full
    event_queue: EventDoubleBuffer = EventDoubleBuffer(size=1)
    event_queue.put(1)
    assert event_queue.event.is_set() == True


def test_put_2() -> None:
    # test the case when buffer is full
    event_queue: EventDoubleBuffer = EventDoubleBuffer(size=1)
    test_data: List[int] = [1, 2, 3]

    for data in test_data:
        event_queue.put(data)
    assert event_queue.event.is_set() == True


def test_get_1() -> None:
    # test the case when buffer is not full
    event_queue: EventDoubleBuffer = EventDoubleBuffer(size=1)
    expected_data: int = 1
    event_queue.put(expected_data)
    assert event_queue.get() == expected_data
    assert event_queue.event.is_set() == False


def test_get_2() -> None:
    # test the case when buffer is full
    event_queue: EventDoubleBuffer = EventDoubleBuffer(size=1)
    test_data: List[int] = [1, 2, 3]
    expected_data: int = 3
    for data in test_data:
        event_queue.put(data)

    assert event_queue.event.is_set() == True
    res = event_queue.get()
    assert res == expected_data
    assert event_queue.event.is_set() == False


def test_get_3() -> None:
    # test the case when buffer is completely empty
    event_queue: EventDoubleBuffer = EventDoubleBuffer(size=1)
    assert event_queue.get() == None
    assert event_queue.event.is_set() == False


def test_get_4() -> None:
    # test the case when buffer is empty but have data previously
    event_queue: EventDoubleBuffer = EventDoubleBuffer(size=1)
    test_data: List[int] = [1, 2, 3]
    expected_data = None

    for data in test_data:
        event_queue.put(data)
        event_queue.get()
    assert event_queue.get() == expected_data
    assert event_queue.event.is_set() == False
