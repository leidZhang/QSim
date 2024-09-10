import math
from abc import ABC
from typing import List, Tuple

from qvl.qlabs import QuanserInteractiveLabs
from qvl.walls import QLabsWalls
from qvl.crosswalk import QLabsCrosswalk
from qvl.traffic_light import QLabsTrafficLight
from qvl.stop_sign import QLabsStopSign
from qvl.flooring import QLabsFlooring
from qvl.qcar import QLabsQCar
from qvl.basic_shape import QLabsBasicShape

from core.roadmap.constants import ACC_X_OFFSET, ACC_Y_OFFSET


class MapBuilder(ABC):
    """
    The abstract Builder class responsible for building the map

    Attributes:
    - qlab: QuanserInteractiveLabs: The QuanserInteractiveLabs object
    - scale: float: The scale of the map
    - offsets: Tuple[float]: The offsets of the map
    """

    def __init__(self, qlab: QuanserInteractiveLabs, offsets: List[float] = [0.0, 0.0]) -> None:
        """
        Initializes the MapBuilder object

        Parameters:
        - qlab: QuanserInteractiveLabs: The QuanserInteractiveLabs object
        - offsets: Tuple[float]: The offsets of the map
        """
        self.qlab: QuanserInteractiveLabs = qlab
        self.scale: float = [0.1, 0.1, 0.1]
        self.offsets: Tuple[float] = offsets
        # add the acc offsets
        self.offsets[0] += ACC_X_OFFSET
        self.offsets[1] += ACC_Y_OFFSET
        # destroy the existing actors
        self.qlab.destroy_all_spawned_actors()

    def build_walls(self) -> None:
        """
        Builds the walls of the map

        Returns:
        - None
        """
        walls: QLabsWalls = QLabsWalls(self.qlab)
        walls.set_enable_dynamics(False)
        for y in range (5):
            walls.spawn_degrees(
                location=[-2.4 + self.offsets[0], (-y * 1.0) + 2.55 + self.offsets[1], 0.001],
                rotation=[0, 0, 0]
            )

        for x in range (5):
            walls.spawn_degrees(
                location=[-1.9 + x + self.offsets[0], 3.05 + self.offsets[1], 0.001],
                rotation=[0, 0, 90]
            )

        for y in range (6):
            walls.spawn_degrees(
                location=[2.4 + self.offsets[0], (-y * 1.0) + 2.55 + self.offsets[1], 0.001],
                rotation=[0, 0, 0]
            )

        for x in range (5):
            walls.spawn_degrees(
                location=[-1.9 + x + self.offsets[0], -3.05 + self.offsets[1], 0.001],
                rotation=[0, 0, 90]
            )

        walls.spawn_degrees(
            location=[-2.03 + self.offsets[0], -2.275 + self.offsets[1], 0.001],
            rotation=[0, 0, 48]
        )
        walls.spawn_degrees(
            location=[-1.575 + self.offsets[0], -2.7 + self.offsets[1], 0.001],
            rotation=[0, 0, 48]
        )

    def build_floor(self) -> None:
        """
        Builds the floor of the map

        Returns:
        - None
        """
        flooring: QLabsFlooring = QLabsFlooring(self.qlab)
        flooring.spawn(location=[self.offsets[0], self.offsets[1], 0.0], rotation=[0, 0, -math.pi/2])

    def build_crosswalk(self) -> None:
        """
        Builds the crosswalk of the map

        Returns:
        - None
        """
        crosswalk: QLabsCrosswalk = QLabsCrosswalk(self.qlab)
        crosswalk.spawn_degrees (location =[-2 + self.offsets[0], -1.475 + self.offsets[1], 0.01],
                rotation=[0,0,0], scale = [0.1,0.1,0.075],
                configuration = 0)
        spline: QLabsBasicShape = QLabsBasicShape(self.qlab)
        spline.spawn_degrees([2.05 + self.offsets[0], -1.5 + self.offsets[1], 0.01], [0, 0, 0], [0.27, 0.02, 0.001], False)
        spline.spawn_degrees([-2.075 + self.offsets[0], self.offsets[1], 0.01], [0, 0, 0], [0.27, 0.02, 0.001], False)


class GeneralMapBuilder(MapBuilder):
    """
    The Builder class responsible for building more general map, the actor will be spawned at the
    position that outside the roadmap at first
    """

    def build_car_actor(self, actor_id: int = 0) -> QLabsQCar:
        """
        Builds the car actor

        Parameters:
        - actor_id: int: The actor id
        """
        car: QLabsQCar = QLabsQCar(self.qlab)
        return car

    def build_traffic_light_actor(self, actor_id: int = 1) -> QLabsTrafficLight:
        """
        Builds the traffic light actor

        Parameters:
        - actor_id: int: The actor id
        """
        traffic_light: QLabsTrafficLight = QLabsTrafficLight(self.qlab)
        return traffic_light

    def build_stop_sign_actor(self, actor_id: int = 0) -> QLabsStopSign:
        """
        Builds the stop sign actor

        Parameters:
        - actor_id: int: The actor id
        """
        stop_sign: QLabsStopSign = QLabsStopSign(self.qlab)
        return stop_sign
