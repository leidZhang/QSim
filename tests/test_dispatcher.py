from copy import deepcopy
from typing import List, Tuple, Dict, Set
from queue import Queue

import pytest
import numpy as np

from core.roadmap.dispatcher import TaskDispacher, OneTimeTaskGenerator
from core.roadmap.constants import ACC_GRAPH_RIGHT


def test_execute() -> None:
    start_node: int = 4
    data_queue: Queue = Queue()

    dispatcher: TaskDispacher = TaskDispacher(start_node=start_node)
    counter: int = 100
    while counter > 0:
        dispatcher.execute(data_queue)
        counter -= 1

    while not data_queue.empty():
        data: Tuple[List[int], np.ndarray] = data_queue.get()
        last_index: int = data[0][-1] # we will check the node sequence
        assert len(data[0]) >= 5 # the minimum length of a task
        assert len(data[0]) <= 15 # the maximum length of a task
        assert len(ACC_GRAPH_RIGHT[last_index]) == 1 # the last node has only one next node


def test_get_task() -> None:
    start_node: int = 4
    dispatcher: OneTimeTaskGenerator = OneTimeTaskGenerator()
    data: Tuple[List[int], np.ndarray] = dispatcher.get_task(start_node_index=start_node)
    last_index: int = data[0][-1] # we will check the node sequence
    assert len(data[0]) >= 3 # the minimum length of a task
    assert len(data[0]) <= 15 # the maximum length of a task