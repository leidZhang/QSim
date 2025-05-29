import time
from typing import List
from multiprocessing import Queue, Process

from hitl import render_raster_map
from generator.generator import run_generator


if __name__ == "__main__":
    run_generator()