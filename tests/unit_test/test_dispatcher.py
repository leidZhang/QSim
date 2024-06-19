from copy import deepcopy
from typing import List, Dict, Set
from queue import Queue

import pytest

from core.roadmap.dispatcher import TaskDispacher
from core.roadmap.constants import ACC_GRAPH_RIGHT


@pytest.mark.unit_test
def test_dispatcher_1() -> None:
    start_node: int = 4
    data_queue: Queue = Queue()

    dispatcher: TaskDispacher = TaskDispacher(start_node=start_node)
    counter: int = 100
    while counter > 0:
        dispatcher.execute(data_queue)
        counter -= 1

    while data_queue.qsize() > 1: # the last one may not be a valid task
        node_sequence: List[int] = data_queue.get()
        last_index: int = node_sequence[-1]
        assert len(node_sequence) >= 5 # the minimum length of a task
        assert len(node_sequence) <= 15 # the maximum length of a task
        assert len(ACC_GRAPH_RIGHT[last_index]) == 1 # the last node has only one next node
