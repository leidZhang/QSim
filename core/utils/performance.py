import sys
import time
from queue import Queue
from typing import Union
from multiprocessing import Queue as MPQueue

import matplotlib.pyplot as plt


def mock_delay(start: float, delay: float) -> None:
    end: float = time.time() - start
    time.sleep(max(0, delay - end))


def skip_delay(*args) -> None:
    pass


def elapsed_time(start_time: float) -> float:
    return time.time() - start_time


def realtime_message_output(message: str) -> None:
    sys.stdout.write(f'\r{message}')
    sys.stdout.flush()


def wait_for_empty_queue_space(data_queue: Union[Queue, MPQueue], wait_time: float = 0.1) -> None:
    while data_queue.full():
        time.sleep(wait_time)


def plot_data_in_dict(data_lists: dict, title: str, x_label: str, y_label: str) -> None:
    for label, data_list in data_lists.items():
        x_axis: list = list(range(len(data_list)))
        y_axis: list = data_list
        plt.plot(x_axis, y_axis, label=label)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

    plt.savefig(f'output/{title}.jpg')

skip = skip_delay
