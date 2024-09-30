import time
from typing import List
from multiprocessing import Queue, Process

from hitl import render_raster_map
from generator.generator import run_generator


if __name__ == "__main__":
    raster_info_queue = Queue()
    # pool: List[Process] = []
    # pool.append(Process(target=render_raster_map, args=(raster_info_queue,)))
    # for process in pool:
    #     process.start()

    # time.sleep(5)
    run_generator(raster_info_queue)
    # for process in pool:
    #     process.join()