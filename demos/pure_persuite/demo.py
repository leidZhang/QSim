import os
import time

from qvl.qlabs import QuanserInteractiveLabs

from core.roadmap import ACCDirector
from demos.override_demo import prepare_map_info
from .car import PPCar

def generate_car(actor_info) -> None:
    actor_info[0].spawn_id(
        actorNumber=0,
        location=actor_info[1],
        rotation=actor_info[2],
        scale=[.1, .1, .1],
        configuration=0,
        waitForConfirmation=True
    )

def run_pure_pursuite() -> None:
    qlabs: QuanserInteractiveLabs = QuanserInteractiveLabs()
    qlabs.open("localhost")
    director: ACCDirector = ACCDirector(qlabs)

    init_pos, waypoint_sequence = prepare_map_info(node_id=24)
    actors: dict = director.build_map(init_pos)
    generate_car(actors["car"])
    car: PPCar = PPCar(qlabs=qlabs, waypoint_sequence=waypoint_sequence)
    car.setup()

    done = False
    while not done:
        try:
            start = time.time()
            car.execute()
            # print(f"Run with {1 / (time.time() - start)} Hz")
        except IndexError:
            print("Reached the goal!")
            car.halt_car()
            done = True

    qlabs.close()