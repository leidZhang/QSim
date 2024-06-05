from typing import Tuple
# quanser imports
from qvl.qlabs import QuanserInteractiveLabs
from qvl.real_time import QLabsRealTime
from qvl.qcar import QLabsQCar
import pal.resources.rtmodels as rtmodels
# custom imports
from .builder import ACCMapBuilder


class ACCDirector:
    """
    The Director class responsible for directing the building of the map for the ACC2024 competition
    """

    def __init__(self, qlabs: QuanserInteractiveLabs, offsets: Tuple[float]) -> None:
        """
        Initializes the ACCDirector object

        Parameters:
        - qlabs: QuanserInteractiveLabs: The QuanserInteractiveLabs object
        """
        self.qlabs: QuanserInteractiveLabs = qlabs
        self.builder: ACCMapBuilder = ACCMapBuilder(self.qlabs, offsets)

    def build_map(self) -> dict:
        """
        Builds the map for the competition

        Parameters:
        - position: list: The position of the car

        Returns:
        - dict: The dictionary containing the actors
        """
        self.qlabs.destroy_all_spawned_actors()
        QLabsRealTime().terminate_all_real_time_models()
        self.builder.build_floor()
        self.builder.build_walls()
        stop_signs: list = self.builder.build_stop_sign()
        self.builder.build_crosswalk()
        traffic_lights: list = self.builder.build_traffic_light()
        car: QLabsQCar = self.builder.preapare_car_spawn()
        QLabsRealTime().start_real_time_model(rtmodels.QCAR_STUDIO)
        return {
            "stop_signs": stop_signs,
            "traffic_lights": traffic_lights,
            "cars": [car]
        }
