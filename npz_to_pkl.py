# python imports
import os
import sys
import time
from multiprocessing import Process

import cv2

from core.qcar.sensor import VirtualRGBDCamera, VirtualCSICamera
from tests.traditional.test import test_master_slave
from tests.cam_threads.test import test_camera_threads
from core.utils.executions import BaseProcessExec
from Reinformer.utils import test_reinformer_util


if __name__ == "__main__":
    # pytest for unit tests
    #python_path: str = sys.executable
    #test_command: str = f"{python_path} -m pytest tests/unit_test/"
    #os.system(test_command)
    test_reinformer_util()
    # test_master_slave()