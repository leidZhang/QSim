# python imports
import os
import sys
import time

import cv2

from core.qcar.sensor import VirtualRGBDCamera, VirtualCSICamera
from tests.performance_environment import prepare_test_environment, destroy_map
from tests.sep_algs.test_sep_algs import test_sep_algs

# pytest
# python_path: str = sys.executable
# test_command: str = f"{python_path} -m pytest tests/unit_test/"
# os.system(test_command)

if __name__ == '__main__':
    assert destroy_map() != -1, "Failed to destroy the map."
    prepare_test_environment(node_id=24)
    test_sep_algs()
    print("Test completed successfully.")

    
            