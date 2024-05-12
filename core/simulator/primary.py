import time

import numpy as np
from typing import List, Dict

from qvl.qlabs import QuanserInteractiveLabs
from qvl.traffic_light import QLabsTrafficLight
from qvl.qcar import QLabsQCar
from qvl.actor import QLabsActor
from qvl.real_time import QLabsRealTime
import pal.resources.rtmodels as rtmodels

from core.roadmap import ACCDirector
from core.roadmap.constants import ACC_X_OFFSET, ACC_Y_OFFSET
from .constants import QCAR_ACTOR_ID
from .monitor import Monitor


class Simulator:
    @classmethod
    def build_map(self, qcar_pos: list, qcar_view: int = 6) -> None:
        ...

    @classmethod
    def reset_map(self, qcar_pos: list, qcar_view: int = 6) -> None:
        ...


class QLabSimulator(Simulator):
    def __init__(self, dt:float = 0.05) -> None:
        """
        Initializes the QLabSimulator object

        Parameters:
        - dt: float: The time step of the simulation
        """
        self.dt: float = dt
        self.qlabs: QuanserInteractiveLabs = QuanserInteractiveLabs()
        self.qlabs.open("localhost")
        self.actors: Dict[str, QLabsActor] = {}
        self.monitors: Dict[str, Monitor] = {
            'car': Monitor(QCAR_ACTOR_ID, 0, dt=self.dt),
        }
        self.regions: Dict[str, np.ndarray] = {
            'stop_signs': None,
            'traffic_lights': None
        }

    def init_actor_states(self) -> None:
        """
        Initialize the monitor state for each actors
        """
        for _, monitor in self.monitors.items():
            monitor.get_position(self.qlabs)

    def render_map(self, qcar_pos: list, qcar_view: int = 6) -> None:
        """
        Renders the map for the simulation

        Parameters:
        - qcar_pos: list: The position of the car
        - qcar_view: int: The view of the car
        """
        director: ACCDirector = ACCDirector(self.qlabs)
        self.actors = director.build_map(qcar_pos)
        self.actors['car'][0].possess(qcar_view)
        self.set_regions()
        time.sleep(2) # cool down time for the car to spawn
        # self.init_actor_states()

    def reset_map(self, qcar_view: int = 6) -> None:
        """
        Resets the actors in the map

        Parameters:
        - qcar_view: int: The view of the car
        """
        QLabsRealTime().terminate_all_real_time_models()
        # delete the old car
        car: QLabsQCar = self.actors['car'][0]
        car.destroy()
        # reset traffic light states
        traffic_lights: List[QLabsTrafficLight] = self.actors['traffic_lights']
        traffic_lights[0].set_state(QLabsTrafficLight.STATE_RED)
        traffic_lights[1].set_state(QLabsTrafficLight.STATE_GREEN)
        # spawn a new car
        location = self.actors['car'][1]
        orientation = self.actors['car'][2]
        car.spawn_id(
            actorNumber=0,
            location=location,
            rotation=orientation,
            scale=[.1, .1, .1],
            configuration=0,
            waitForConfirmation=True
        )
        car.possess(qcar_view)
        time.sleep(1)
        QLabsRealTime().start_real_time_model(rtmodels.QCAR_STUDIO)
        time.sleep(2) # wait for the state to change
        self.init_actor_states()

    def set_waypoint_sequence(self, waypoints: np.ndarray) -> None:
        """
        Set the waypoint sequence for the simulation

        Parameters:
        - waypoints: np.ndarray: The waypoints for the simulation
        """
        self.waypoints: np.ndarray = waypoints

    def get_actor_state(self, actor_name: str) -> np.ndarray:
        """
        Get the state of the actor

        Parameters:
        - actor_name: str: The name of the actor
        """
        self.monitors[actor_name].get_state(self.qlabs)
        return self.monitors[actor_name].state

    def set_regions(self) -> None:
        # set the regions for the stop signs and traffic lights
        self.regions['stop_signs'] = np.stack([
            np.array([
                [2.1 + ACC_X_OFFSET - (0.25 / 2), 1.15 + ACC_Y_OFFSET - (0.45 / 2)],
                [2.1 + ACC_X_OFFSET + (0.25 / 2), 1.15 + ACC_Y_OFFSET + (0.45 / 2)]],
                dtype=np.float32
            ),
            np.array([
                [-0.95 + ACC_X_OFFSET - (0.45 / 2), 2.75 + ACC_Y_OFFSET - (0.25 / 2)],
                [-0.95 + ACC_X_OFFSET + (0.45 / 2), 2.75 + ACC_Y_OFFSET + (0.25 / 2)]],
                dtype=np.float32
            )
        ])
        self.regions['traffic_lights'] = np.stack([
            np.array([
                [-2.075 + ACC_X_OFFSET - (0.45 / 2), 0.35 + ACC_Y_OFFSET - (0.25 / 2)],
                [-2.075 + ACC_X_OFFSET + (0.45 / 2), 0.35 + ACC_Y_OFFSET + (0.25 / 2)]],
                dtype=np.float32
            ),
            np.array([
                [2.1 + ACC_X_OFFSET - (0.25 / 2), -1.85 + ACC_Y_OFFSET - (0.45 / 2)],
                [2.1 + ACC_X_OFFSET + (0.25 / 2), -1.85 + ACC_Y_OFFSET + (0.45 / 2)]],
                dtype=np.float32
            )
        ])
