from multiprocessing import Queue
from typing import Callable, Any

import cv2

from core.roadmap import ACCRoadMap
from .hitl_policy import KeyboardPolicy
from .hitl_raster_map import *


def run_keyboard_policy(callback: Callable = lambda *args: None, *args: Any) -> None:
    # mainly for test purpose
    while True:
        hitl: KeyboardPolicy = KeyboardPolicy()
        accelerate, break_pedal = hitl.execute()
        print(accelerate, break_pedal)
        callback(accelerate, break_pedal, *args)


def render_raster_map(raster_queue: Queue) -> None:
    try:
        # run in sub-process
        roadmap: ACCRoadMap = ACCRoadMap()
        renderer: HITLRasterMap = HITLRasterMap(
            road_map=roadmap,
            map_size=CR_MAP_SIZE,
            map_params=CR_MAP_PARAMS
        )

        while True:
            map_info: Any = raster_queue.get()
            ego_state, hazard_states, waypoint_list = map_info
            raster_map, _, _ = renderer.draw_map(ego_state, hazard_states, waypoint_list)
            cv2.imshow("HITL Raster Map", raster_map)
            cv2.waitKey(1)
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
