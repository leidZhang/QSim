from typing import Tuple, Dict, Any, List
# quanser imports
from qvl.qlabs import QuanserInteractiveLabs
from qvl.real_time import QLabsRealTime
from qvl.qcar import QLabsQCar
from qvl.actor import QLabsActor
import pal.resources.rtmodels as rtmodels
# custom imports
from .builder import MapBuilder, GeneralMapBuilder


class GeneralDirector:
    def __init__(
        self,
        builder: GeneralMapBuilder,
        build_walls: bool = True,
    ) -> None:
        self.builder: GeneralMapBuilder = builder
        self.build_walls: bool = build_walls

    def build_map(self, configs: Dict[str, Dict[int, Any]] = None) -> Dict[str, List[QLabsActor]]:
        """
        Builds the map for the competition

        Parameters:
        - configs (Dict[str, Dict[int, Any]]): The dictionary containing the configurations for the actors

        Returns:
        - Dict[str, List[QLabsActor]]: The dictionary containing the actors
        """
        actors: Dict[str, List[QLabsActor]] = {
            "cars": [],
            "stop_signs": [],
            "traffic_lights": [],
        }

        # terminate all real-time models
        QLabsRealTime().terminate_all_real_time_models()

        # build the common parts of the map
        self.builder.build_floor()
        self.builder.build_crosswalk()
        self.builder.build_walls() if self.build_walls else lambda: None
        # build the actors based on the configs
        if configs is not None:
            for key in configs["cars"].keys():
                car = self.builder.build_car_actor(key)
                actors["cars"].append(car)
            for key in configs["stop_signs"].keys():
                stop_sign = self.builder.build_stop_sign_actor(key)
                actors["stop_signs"].append(stop_sign)
            for key in configs["traffic_lights"].keys():
                traffic_light = self.builder.build_traffic_light_actor(key)
                actors["traffic_lights"].append(traffic_light)

        QLabsRealTime().start_real_time_model(rtmodels.QCAR_STUDIO)

        return actors, self.builder.qlab
