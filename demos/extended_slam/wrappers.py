from typing import List, Tuple, Any, Set

import numpy as np

from core.qcar import LidarSLAM, VirtualRGBDCamera
from core.control.automata import BaseState, EventDrivenDFA
from core.utils.ipc_utils import DoubleBuffer
from demos.traditional.observe import DecisionMaker
from .modules import Event, SLAMCar
from .triggers import RollingAverageTrigger, StopSignTrigger, TrafficLightTrigger
from .states import MovingState, StopSignState, RedLightState, FinalState


class SLAMCarWrapper:
    def __init__(self, events: List[Event]) -> None:
        super().__init__()
        self.car: SLAMCar = SLAMCar(events)
        self.events: List[Event] = events

    def setup_thread(self) -> None:
        init_state_index: int = 0
        final_states: Set[int] = {3}
        states: List[BaseState] = [
            MovingState(self.car, self.events),
            StopSignState(self.car, self.events),
            RedLightState(self.car, self.events),
            FinalState(self.car)
        ]

        self.fsm: EventDrivenDFA = EventDrivenDFA(states, final_states, init_state_index)

    def execute(self, request: DoubleBuffer) -> None:
        self.fsm.execute(request)


class SLAMObserverWrapper: 
    def __init__(self, events) -> None:
        self.triggers: List[RollingAverageTrigger] = [
            StopSignTrigger(events[1], 0.5, 3), # for stop sign detection
            TrafficLightTrigger(events[2], 0.3, 10) # for traffic light detection
        ]
        self.observer: DecisionMaker = DecisionMaker(
            classic_traffic_pipeline=True,
            network_class=None,
            output_postprocess=lambda x: x.argmax().item(),
            weights_file=None,
            device='cuda'
        )

    def _handle_triggers(self, res: int) -> None:
        for trigger in self.triggers:
            trigger(res)
            if trigger.event.is_set():
                return

    def execute(self, request: DoubleBuffer) -> None:
        # get the latest image from the multiprocessing queue
        image: np.ndarray = request.get()
        if image is not None:
            res: int = self.observer(image)
            self._handle_triggers(res)


class SensorsWrapper: # put all sensors in one thread
    def __init__(self) -> None:
        self.gps: LidarSLAM = LidarSLAM([0, 2, np.pi / 2])
        self.rgbd: VirtualRGBDCamera = VirtualRGBDCamera()

    def execute(self, control_queue: DoubleBuffer, detection_queue: DoubleBuffer) -> None:
        self.gps.readGPS()
        state: np.ndarray = np.array([
            self.gps.position[0],
            self.gps.position[1],
            self.gps.orientation[2],
            0, 0, 0
        ])
        control_queue.put(state.copy())
        image: np.ndarray = self.rgbd.read_rgb_image()
        if image is not None:
            detection_queue.put(image.copy())


class GPSWrapper: # put GPS in one thread
    def __init__(self) -> None:
        self.gps: LidarSLAM = LidarSLAM([0, 2, np.pi / 2])

    def execute(
        self, 
        control_queue: DoubleBuffer, 
    ) -> None:
        self.gps.readGPS()
        state: np.ndarray = np.array([
            self.gps.position[0],
            self.gps.position[1],
            self.gps.orientation[2],
            0, 0, 0
        ])
        control_queue.put(state.copy())


class RGBDWrapper: # put RGBD in one thread
    def __init__(self) -> None:
        self.camera: VirtualRGBDCamera = VirtualRGBDCamera()

    def execute(
        self, 
        detection_queue: DoubleBuffer, 
    ) -> None:
        image: np.ndarray = self.camera.read_rgb_image()
        if image is not None:
            detection_queue.put(image.copy())
