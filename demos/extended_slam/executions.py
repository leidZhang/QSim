import numpy as np

from core.utils.executions import BaseThreadExec, BaseProcessExec
from core.utils.ipc_utils import DoubleBuffer
from demos.performance_environment import prepare_test_environment
from .modules import WaypointProcessor
from .wrappers import SensorsWrapper, SLAMCarWrapper, SLAMObserverWrapper


class WaypointProcessorExec(BaseProcessExec):
    def create_instance(self, request_queue: DoubleBuffer) -> WaypointProcessor:
        waypoints: np.ndarray = prepare_test_environment(node_id=10)
        waypoint_processor: WaypointProcessor = WaypointProcessor(waypoints=waypoints)
        waypoint_processor.setup(0, request_queue)
        return waypoint_processor
        
    def run_process(self, request_queue: DoubleBuffer, response_queue: DoubleBuffer) -> None:
        instance: WaypointProcessor = self.create_instance(request_queue)
        while not self.done.is_set():
            instance.execute(request_queue, response_queue)
        self.final()


class SLAMObserverExec(BaseProcessExec):
    def create_instance(self) -> SLAMObserverWrapper:
        return SLAMObserverWrapper()


class SensorThreadExec(BaseThreadExec):
    def __init__(self) -> None:
        super().__init__()
        self.sensors: SensorsWrapper = SensorsWrapper()

    def execute(
        self, 
        control_queue: DoubleBuffer, 
        detection_queue: DoubleBuffer
    ) -> None:
        self.sensors.execute(control_queue, detection_queue)


class SLAMCarThreadExec(BaseThreadExec):
    def __init__(self, events: list) -> None:
        super().__init__()
        self.car: SLAMCarWrapper = SLAMCarWrapper(events)

    def execute(self, request: DoubleBuffer) -> None:
        self.car.execute(request)
