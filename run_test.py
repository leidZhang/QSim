# python imports
import os
import sys
import time

import cv2

from core.qcar.sensor import VirtualRGBDCamera, VirtualCSICamera
from tests.performance_environment import prepare_test_environment, destroy_map
from tests.performance_environment import prepare_test_environment_waypoint
from tests.sep_algs.test_sep_algs import test_sep_algs
from tests.edge_finder.test_edge_finder import test_edge_finder

# pytest
if __name__ == "__main__": 
    python_path: str = sys.executable
    test_command: str = f"{python_path} -m pytest tests/unit_test/"
    os.system(test_command)
    