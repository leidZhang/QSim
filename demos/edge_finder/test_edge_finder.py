import time
from typing import Tuple

import cv2
import numpy as np

from core.qcar.sensor import VirtualCSICamera
from core.control.edge_finder import TraditionalEdgeFinder, EdgeFinder, NoContourException
from core.utils.performance import realtime_message_output

def handle_iteration(edge_finder: EdgeFinder, image: np.ndarray) -> Tuple[float, float]:
    try: 
        if image is not None:
            # 
            result: tuple = edge_finder.execute(image)
            # realtime_message_output(f"Result: {result}")
            # realtime_message_output(f"Frequency for iteration: {1 /(time.time() - iter_start)}Hz")
            return result
    except NoContourException:
        pass


def test_edge_finder() -> None:
    start_time: float = time.time()
    camera: VirtualCSICamera = VirtualCSICamera()
    edge_finder: EdgeFinder = TraditionalEdgeFinder()

    counter: int = 0
    while time.time() - start_time < 100: 
        iter_start: float = time.time()
        image: np.ndarray = camera.read_image()
        result: Tuple[float, float] = handle_iteration(edge_finder, image)
        iter_end: float = time.time() - iter_start
        # if counter % 5 == 0 and iter_end != 0:
        #     realtime_message_output(f"Result: {result}, Frequency for iteration: {1 / iter_end}Hz {' ' * 10}")
        cv2.waitKey(1)
        counter += 1
            
    print("Edge finder test completed.")