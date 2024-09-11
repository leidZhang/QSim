import time
from copy import deepcopy

from qvl.qlabs import QuanserInteractiveLabs
from core.roadmap import ACCRoadMap
from core.environment.builder import GeneralMapBuilder
from core.environment.director import GeneralDirector
from core.environment.constants import FULL_CONFIG
from core.environment.simulator import QLabSimulator
from core.policies import PurePursuiteAdaptor
from .environment import CrossRoadEnvironment, OnlineQLabEnv


def run_cross_road_env_demo():
    print("This demo shows the cross-road environment in action.")
    roadmap: ACCRoadMap = ACCRoadMap()
    qlabs: QuanserInteractiveLabs = QuanserInteractiveLabs()
    qlabs.open("localhost")

    print("Building the map...")
    config: dict = deepcopy(FULL_CONFIG)
    config["traffic_lights"] = {} # delete the traffic lights
    for i in range(1, 4):
        config["cars"][i] = None # add a hazardous car to the map
    builder: GeneralMapBuilder = GeneralMapBuilder(qlabs)
    director: GeneralDirector = GeneralDirector(builder)
    sim: QLabSimulator = QLabSimulator([0, 0], False)
    sim.render_map(director, config)

    print("Starting the environment...")
    env: OnlineQLabEnv = CrossRoadEnvironment(sim, roadmap, privileged=True)
    env.set_ego_policy(PurePursuiteAdaptor())
    env.set_hazard_policy(PurePursuiteAdaptor())

    for i in range(500):
        print(f"Starting episode {i}...")
        env.reset() # use this to set the agents to their initial positions
        for _ in range(50):
            _, _, done, _ = env.step()
            if done:
                print(f"Episode {i + 1} complete")
                break
        env.stop_all_agents()
        time.sleep(2)
    print("Demo complete")