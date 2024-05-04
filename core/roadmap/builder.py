import math
from typing import List

from qvl.qlabs import QuanserInteractiveLabs
from qvl.walls import QLabsWalls
from qvl.crosswalk import QLabsCrosswalk
from qvl.traffic_light import QLabsTrafficLight
from qvl.stop_sign import QLabsStopSign
from qvl.flooring import QLabsFlooring
from qvl.qcar import QLabsQCar
from qvl.basic_shape import QLabsBasicShape

from .constants import ACC_X_OFFSET, ACC_Y_OFFSET

class ACCMapBuilder:
    """
    The Builder class responsible for building the map for the ACC2024 competition

    Attributes:
    - qlab: QuanserInteractiveLabs: The QuanserInteractiveLabs object
    - scale: float: The scale of the map

    Methods:
    - build_walls: Builds the walls of the map
    - build_floor: Builds the floor of the map
    - build_stop_sign: Builds the stop signs of the map
    - build_crosswalk: Builds the crosswalk of the map
    - build_traffic_light: Builds the traffic lights of the map
    - build_car: Builds the car for the map
    """

    def __init__(self, qlab: QuanserInteractiveLabs) -> None:
        """
        Initializes the ACCMapBuilder object

        Parameters:
        - qlab: QuanserInteractiveLabs: The QuanserInteractiveLabs object
        """
        self.qlab: QuanserInteractiveLabs = qlab
        self.scale: float = [0.1, 0.1, 0.1]

    def build_walls(self) -> None:
        """
        Builds the walls of the map
        """
        walls: QLabsWalls = QLabsWalls(self.qlab)
        walls.set_enable_dynamics(False)
        for y in range (5):
            walls.spawn_degrees(location=[-2.4 + ACC_X_OFFSET, (-y*1.0) + 2.55 + ACC_Y_OFFSET, 0.001], rotation=[0, 0, 0])

        for x in range (5):
            walls.spawn_degrees(location=[-1.9+x + ACC_X_OFFSET, 3.05 + ACC_Y_OFFSET, 0.001], rotation=[0, 0, 90])

        for y in range (6):
            walls.spawn_degrees(location=[2.4+ ACC_X_OFFSET, (-y*1.0) + 2.55 + ACC_Y_OFFSET, 0.001], rotation=[0, 0, 0])

        for x in range (5):
            walls.spawn_degrees(location=[-1.9+x+ ACC_X_OFFSET, -3.05 + ACC_Y_OFFSET, 0.001], rotation=[0, 0, 90])

        walls.spawn_degrees(location=[-2.03 + ACC_X_OFFSET, -2.275 + ACC_Y_OFFSET, 0.001], rotation=[0, 0, 48])
        walls.spawn_degrees(location=[-1.575+ ACC_X_OFFSET, -2.7 + ACC_Y_OFFSET, 0.001], rotation=[0, 0, 48])

    def build_floor(self) -> None:
        """
        Builds the floor of the map
        """
        flooring: QLabsFlooring = QLabsFlooring(self.qlab)
        flooring.spawn(location=[ACC_X_OFFSET, ACC_Y_OFFSET, 0.0], rotation=[0, 0, -math.pi/2])

    def build_stop_sign(self) -> None:
        """
        Builds the stop signs of the map
        """
        stop_signs: list = []
        stop_sign: QLabsStopSign = QLabsStopSign(self.qlab)
        stop_sign_1 = stop_sign.spawn_degrees([2.25 + ACC_X_OFFSET, 1.5 + ACC_Y_OFFSET, 0.05], [0, 0, -90], [0.1, 0.1, 0.1], False)[1]
        stop_sign_2 = stop_sign.spawn_degrees([-1.3 + ACC_X_OFFSET, 2.9 + ACC_Y_OFFSET, 0.05], [0, 0, -15], [0.1, 0.1, 0.1], False)[1]
        stop_signs.append(stop_sign_1)
        stop_signs.append(stop_sign_2)
        return stop_signs

    def build_crosswalk(self) -> None:
        """
        Builds the crosswalk of the map
        """
        crosswalk: QLabsCrosswalk = QLabsCrosswalk(self.qlab)
        crosswalk.spawn_degrees (location =[-2 + ACC_X_OFFSET, -1.475 + ACC_Y_OFFSET, 0.01],
                rotation=[0,0,0], scale = [0.1,0.1,0.075],
                configuration = 0)
        spline: QLabsBasicShape = QLabsBasicShape(self.qlab)
        spline.spawn_degrees([2.05 + ACC_X_OFFSET, -1.5 + ACC_Y_OFFSET, 0.01], [0, 0, 0], [0.27, 0.02, 0.001], False)
        spline.spawn_degrees([-2.075 + ACC_X_OFFSET, ACC_Y_OFFSET, 0.01], [0, 0, 0], [0.27, 0.02, 0.001], False)

    def build_traffic_light(self) -> list:
        """
        Builds the traffic lights of the map
        """
        traffic_lights: List[QLabsTrafficLight] = [QLabsTrafficLight(self.qlab)] * 2
        traffic_lights[0].spawn_degrees([2.3 + ACC_X_OFFSET, ACC_Y_OFFSET, 0], [0, 0, 0], scale=[.1, .1, .1],
                                        configuration=0, waitForConfirmation=True)
        traffic_lights[0].set_state(QLabsTrafficLight.STATE_GREEN)
        traffic_lights[1].spawn_degrees([-2.3 + ACC_X_OFFSET, -1 + ACC_Y_OFFSET, 0], [0, 0, 180],
                                              scale=[.1, .1, .1], configuration=0, waitForConfirmation=True)
        traffic_lights[1].set_state(QLabsTrafficLight.STATE_RED)
        return traffic_lights

    def build_car(self, position: list) -> tuple:
        """
        Spawn the qcar at the specified position of the map

        Parameters:
        - position: list: The position of the car

        Returns:
        - tuple: The car, position, and orientation of the car
        """
        car: QLabsQCar = QLabsQCar(self.qlab)
        basic_shape: QLabsBasicShape = QLabsBasicShape(self.qlab)
        car_position: list = [position[0], position[1], 0.0]
        car_orientation: list = [0, 0, position[2]]

        # car.spawn_id(
        #     actorNumber=0,
        #     location=car_position,
        #     rotation=car_orientation,
        #     scale=[.1, .1, .1],
        #     configuration=0,
        #     waitForConfirmation=True
        # )

        basic_shape.spawn_id_and_parent_with_relative_transform(
            actorNumber=102, location=[1.15, 0, 1.8],
            rotation=[0, 0, 0], scale=[.65, .65, .1],
            configuration=basic_shape.SHAPE_SPHERE,
            parentClassID=car.ID_QCAR,
            parentActorNumber=2,
            parentComponent=1,
            waitForConfirmation=True
        )
        basic_shape.set_material_properties(color=[0.4,0,0], roughness=0.4, metallic=True, waitForConfirmation=True)
        return (car, car_position, car_orientation)