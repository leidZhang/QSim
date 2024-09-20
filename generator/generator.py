import time
from copy import deepcopy
from typing import Any, Dict
from multiprocessing import Queue

from qvl.qlabs import QuanserInteractiveLabs
from core.roadmap import ACCRoadMap
from core.environment.builder import GeneralMapBuilder
from core.environment.director import GeneralDirector
from core.environment.constants import FULL_CONFIG
from core.environment.simulator import QLabSimulator
from core.policies import PurePursuiteAdaptor
from .environment import CrossRoadEnvironment, OnlineQLabEnv


def prepare_cross_road_env() -> CrossRoadEnvironment:
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

    return env


def transform_observation(episode_observation: Dict[str, Any]) -> Dict[str, list]:
    data: Dict[str, list] = {}
    for step_observation in episode_observation:
        for key, value in step_observation.items():
            if key not in data:
                data[key] = []
            data[key].append(value)
    return data


# TODO: Refactor it to a class
def run_generator(raster_queue: Queue):
    env: OnlineQLabEnv = prepare_cross_road_env()
    for i in range(500):
        print(f"Starting episode {i}...")
        episode_reward, episode_observation = 0, []
        # use this to set the agents to their initial positions
        observation, reward, done, _ = env.reset(raster_queue)
        for _ in range(100):
            start: float = time.time()
            observation, reward, done, _ = env.step(raster_queue)
            observation["reward"] = reward
            episode_observation.append(observation)
            episode_reward += reward
            if done:
                break
            time.sleep(max(0, env.dt - (time.time() - start)))
        print(f"Episode {i + 1} complete with reward {episode_reward}")
        env.stop_all_agents()

        # data = transform_observation(episode_observation)
        # print(data.keys())
        time.sleep(2)
    print("Demo complete")