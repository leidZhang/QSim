import time
from abc import abstractmethod
from typing import List, Dict, Tuple

import numpy as np

from qvl.qlabs import QuanserInteractiveLabs
from qvl.traffic_light import QLabsTrafficLight
from qvl.qcar import QLabsQCar
from qvl.actor import QLabsActor
from qvl.real_time import QLabsRealTime
import pal.resources.rtmodels as rtmodels

from core.roadmap import ACCDirector
from core.roadmap.constants import ACC_X_OFFSET, ACC_Y_OFFSET
from core.qcar.vehicle import VirtualCar


class Simulator:
    @abstractmethod
    def render_map(self, *args) -> None:
        ...

    @abstractmethod
    def reset_map(self,  *args) -> None:
        ...


class QLabSimulator(Simulator):
    def __init__(self, offsets: Tuple[float], qcar_id: int = 0) -> None:
        """
        Initializes the QLabSimulator object

        Parameters:
        - dt: float: The time step of the simulation
        """
        self.qcar_id: int = qcar_id
        self.offsets: Tuple[float] = offsets
        self.qlabs: QuanserInteractiveLabs = QuanserInteractiveLabs()
        self.vehicles: Dict[int, VirtualCar] = {}
        self.regions: Dict[str, np.ndarray] = {
            'stop_signs': None,
            'traffic_lights': None
        }

    # def add_vehicle(self, actor_id: int, throttle_coeff: float = 0.3, steering_coeff: float = 0.5) -> None:
    #     return VirtualCar(actor_id, self.dt, self.qlabs, throttle_coeff, steering_coeff)

    def render_map(self) -> None:
        """
        Renders the map for the simulation

        Parameters:
        - qcar_pos: list: The position of the car
        - qcar_view: int: The view of the car
        """
        self.qlabs.open("localhost")
        director: ACCDirector = ACCDirector(self.qlabs, self.offsets)
        self.actors: Dict[str, QLabsActor] = director.build_map()
        self.set_regions()
        time.sleep(2) # cool down time for the car to spawn
        # self.init_actor_states()

    def reset_map(self, location: list, orientation: float, qcar_view: int = 6) -> None:
        """
        Resets the actors in the map

        Parameters:
        - qcar_view: int: The view of the car
        """
        QLabsRealTime().terminate_all_real_time_models()
        # delete the old car
        car: QLabsQCar = self.actors['cars'][0]
        car.destroy()
        # reset traffic light states
        traffic_lights: List[QLabsTrafficLight] = self.actors['traffic_lights']
        traffic_lights[0].set_state(QLabsTrafficLight.STATE_RED)
        traffic_lights[1].set_state(QLabsTrafficLight.STATE_GREEN)
        # spawn a new car
        car.spawn_id(
            actorNumber=self.qcar_id,
            location=location,
            rotation=orientation,
            scale=[.1, .1, .1],
            configuration=0,
            waitForConfirmation=True
        )
        # car.possess(qcar_view)
        time.sleep(1)
        QLabsRealTime().start_real_time_model(rtmodels.QCAR_STUDIO)
        time.sleep(2) # wait for the state to change
        # self.init_actor_states()

    def set_waypoint_sequence(self, waypoints: np.ndarray) -> None:
        """
        Set the waypoint sequence for the simulation

        Parameters:
        - waypoints: np.ndarray: The waypoints for the simulation
        """
        self.waypoints: np.ndarray = waypoints

    def set_regions(self) -> None:
        """
        Set the regions for the stop signs and traffic lights

        Returns:
        - None
        """
        # set the regions for the stop signs and traffic lights
        self.regions['stop_signs'] = np.stack([
            np.array([
                [2.1 + ACC_X_OFFSET - (0.25 / 2) + self.offsets[0], 1.15 + ACC_Y_OFFSET - (0.45 / 2) + self.offsets[1]],
                [2.1 + ACC_X_OFFSET + (0.25 / 2) + self.offsets[0], 1.15 + ACC_Y_OFFSET + (0.45 / 2) + self.offsets[1]]],
                dtype=np.float32
            ),
            np.array([
                [-0.95 + ACC_X_OFFSET - (0.45 / 2) + self.offsets[0], 2.75 + ACC_Y_OFFSET - (0.25 / 2) + self.offsets[1]],
                [-0.95 + ACC_X_OFFSET + (0.45 / 2) + self.offsets[0], 2.75 + ACC_Y_OFFSET + (0.25 / 2) + self.offsets[1]]],
                dtype=np.float32
            )
        ])
        self.regions['traffic_lights'] = np.stack([
            np.array([
                [-2.075 + ACC_X_OFFSET - (0.45 / 2) + self.offsets[0], 0.35 + ACC_Y_OFFSET - (0.25 / 2) + self.offsets[1]],
                [-2.075 + ACC_X_OFFSET + (0.45 / 2) + self.offsets[0], 0.35 + ACC_Y_OFFSET + (0.25 / 2) + self.offsets[1]]],
                dtype=np.float32
            ),
            np.array([
                [2.1 + ACC_X_OFFSET - (0.25 / 2) + self.offsets[0], -1.85 + ACC_Y_OFFSET - (0.45 / 2) + self.offsets[1]],
                [2.1 + ACC_X_OFFSET + (0.25 / 2) + self.offsets[0], -1.85 + ACC_Y_OFFSET + (0.45 / 2) + self.offsets[1]]],
                dtype=np.float32
            )
        ])
