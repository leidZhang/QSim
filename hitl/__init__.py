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


# TODO: Have some problem
def render_raster_map(raster_info_queue: Queue, raster_data_queue: Queue) -> None:
    # try:
    # run in sub-process
    roadmap: ACCRoadMap = ACCRoadMap()
    renderer: HITLRasterMap = HITLRasterMap(
        road_map=roadmap,
        map_size=CR_MAP_SIZE,
        map_params=CR_MAP_PARAMS
    )

    print("Start rendering HITL Raster Map")
    while True:
        raster_maps: List[np.ndarray] = []
        map_info: Any = raster_info_queue.get()
        agent_states, waypoint_list = map_info
        for i in range(len(agent_states)):
            ego_state = agent_states[i]
            hazard_states = agent_states[:i] + agent_states[i+1:]
            raster_map, _, _ = renderer.draw_map(ego_state, hazard_states, waypoint_list)
            raster_maps.append(raster_map)
            cv2.imshow(f"HITL Raster Map {i}", raster_map)
        cv2.waitKey(1)
        raster_data_queue.put(raster_maps.copy())
    # except KeyboardInterrupt:
    #     print("Interrupted by user")
    # except Exception as e:
    #     print(f"Error: {e}")
