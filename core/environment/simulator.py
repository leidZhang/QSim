import time
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any

import numpy as np

from qvl.qlabs import QuanserInteractiveLabs
from qvl.actor import QLabsActor

from .director import GeneralDirector


class Simulator(ABC):
    """
    The Simulator class is an abstract class that defines the interface for the simulator.
    """

    @abstractmethod
    def render_map(self, *args) -> None:
        """
        Renders the map in the simulator. This method is abstract and must be implemented by the subclass.
        """
        ...

    @abstractmethod
    def reset_map(self,  *args) -> None:
        """
        Resets the map in the simulator. This method is abstract and must be implemented by the subclass.
        """
        ...


class QLabSimulator(Simulator):
    """
    QLabSimulator is a class that implements the Simulator interface for the Quanser Interactive Labs simulator.
    It is the standard class for the simulator layer in the Quanser Interactive Labs environment. It is responsible
    for rendering and resetting the map in the simulator.

    Attributes:
    - offsets: Tuple[float]: The offsets for the map
    - random_traffic_objects: bool: A flag to determine if the traffic objects are random
    - qlabs: QuanserInteractiveLabs: The Quanser Interactive Labs object
    - actors: Dict[str, List[QLabsActor]]: The actors in the simulation
    - config: Dict[str, Dict[str, Any]]: The configuration for the actors
    - in_map_objects: Dict[str, List[int]]: The objects in the map during the episode
    """

    def __init__(
        self,
        offsets: Tuple[float],
        random_traffic_objects: bool = False,
    ) -> None:
        """
        The QLabSimulator class constructor, initialize the simulator with the given parameters.

        Parameters:
        - offsets: Tuple[float]: The offsets for the map
        - random_traffic_objects: bool: A flag to determine if the traffic objects are random
        """
        self.offsets: List[float] = offsets
        self.random_traffic_objects = random_traffic_objects

        self.qlabs: QuanserInteractiveLabs = None
        self.actors: Dict[str, List[QLabsActor]] = None
        self.config: Dict[str, Dict[str, Any]] = None
        self.in_map_objects: Dict[str, List[int]] = None

    def render_map(self, director: GeneralDirector, config: Dict[str, Any]) -> None:
        """
        Renders the map in the simulator using the given director and configuration. It will first
        spawn all the actors outside of the main areana.

        Parameters:
        - director(GeneralDirector): The director to build the map
        - config(Dict[str, Any]): The configuration for the actors

        Returns:
        - None
        """
        self.config = config
        self.actors, self.qlabs = director.build_map(configs=config)
        for i, actor_list in enumerate(self.actors.values()):
            for j, actor in enumerate(actor_list):
                location: List[float] = [5, (i+1) * (j+1) * 0.5 + 10, 0]
                actor.spawn_id(actorNumber=j, location=location, scale=[.1, .1, .1])
        time.sleep(2)

    def reset_map(self) -> None:
        """
        Resets the map in the simulator, it will first move all the actors outside of the main areana,
        then move them to the correct location.

        Returns:
        - None
        """
        self.in_map_objects = {}
        # reset car actors
        for i, car in enumerate(self.actors['cars']):
            car.set_transform_and_request_state(
                location=[5, (i+1) * 0.5 + 10, 0],
                rotation=[0, 0, 0],
                enableDynamics=True,
                headlights=False,
                leftTurnSignal=False,
                rightTurnSignal=False,
                reverseSignal=False,
                brakeSignal=False,
                waitForConfirmation=True
            )

        # reset the general actors
        for key, actor_list in self.actors.items():
            if key == 'cars' or len(actor_list) == 0:
                continue

            rand_size: int = random.randint(1, len(actor_list)) if self.random_traffic_objects else len(actor_list)
            indices: List[int] = random.sample(range(len(actor_list)), rand_size)
            # print(key, sorted(indices), "generated")
            self.in_map_objects[key] = {}
            self.in_map_objects[key]["indices"] = sorted(indices)
            self.in_map_objects[key]["offsets"] = {}
            for i, actor in enumerate(actor_list):
                actor.destroy()
                if i not in indices:
                    continue

                pose: np.ndarray = self.config[key][i]['pose']
                x_noise = np.random.normal(scale=0.025) if self.config[key][i]["random"] else 0
                y_noise = np.random.normal(scale=0.025) if self.config[key][i]["random"] else 0
                self.in_map_objects[key]["offsets"][i] = np.array([x_noise, y_noise, 0])
                location: List[float] = [pose[0] + x_noise, pose[1] + y_noise, 0]
                orientation: List[float] = [0, 0, pose[2]]
                actor.spawn_id(actorNumber=i, location=location, rotation=orientation, scale=[.1, .1, .1])

    def set_car_pos(self, actor_id: int, location: List[float], orientation: List[float]) -> None:
        """
        Set the car position in the simulator based on the given location and orientation, and the actor id.

        Parameters:
        - actor_id(int): The actor id
        - location(List[float]): The location of the car
        - orientation(List[float]): The orientation of the car

        Returns:
        - None
        """
        self.actors['cars'][actor_id].set_transform_and_request_state(
            location=location,
            rotation=orientation,
            enableDynamics=True,
            headlights=False,
            leftTurnSignal=False,
            rightTurnSignal=False,
            reverseSignal=False,
            brakeSignal=False,
            waitForConfirmation=True
        )
