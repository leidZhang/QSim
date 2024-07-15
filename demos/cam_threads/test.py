import time
from threading import Thread
from multiprocessing import Process

from demos.performance_environment import destroy_map, prepare_test_environment
from .scripts import run_rgbd_thread, run_csi_thread
from .scripts import run_hardware_process

def test_camera_threads() -> None:
    destroy_map()
    prepare_test_environment(node_id=24)

    rgbd_thread: Thread = Thread(target=run_rgbd_thread)
    csi_thread: Thread = Thread(target=run_csi_thread)
    car_process: Process = Process(target=run_hardware_process)

    rgbd_thread.start()
    csi_thread.start()
    car_process.start()
    rgbd_thread.join()
    csi_thread.join()