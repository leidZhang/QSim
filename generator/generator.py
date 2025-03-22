import time
from copy import deepcopy
from typing import *
from multiprocessing import Queue

import wandb
import numpy as np

from qvl.qlabs import QuanserInteractiveLabs
from core.roadmap import ACCRoadMap
from core.environment import AnomalousEpisodeException
from core.environment.builder import GeneralMapBuilder
from core.environment.director import GeneralDirector
from core.environment.constants import FULL_CONFIG
from core.environment.simulator import QLabSimulator
from core.policies import PurePursuiteAdaptor
from settings import *
from dqn import CompositeDQNPolicy
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
    env: OnlineQLabEnv = CrossRoadEnvironment(sim, roadmap, privileged=True, dt=0.02)
    env.set_ego_policy(PurePursuiteAdaptor(max_lookahead_distance=0.68))
    # env.set_ego_policy(CompositeDQNPolicy())
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


def run_episode(env: CrossRoadEnvironment, i: int, raster_queue: Queue) -> List[Dict[str, Any]]:
    # print(f"Starting episode {i}...")
    episode_reward, episode_observation = 0, []
    # use this to set the agents to their initial positions
    observation, reward, done, _ = env.reset(raster_queue)
    for _ in range(150):
        start: float = time.time()
        observation, reward, done, _ = env.step(raster_queue)
        observation["reward"] = reward
        episode_observation.append(observation)
        episode_reward += reward
        if done:
            break
        time.sleep(max(0, env.dt - (time.time() - start)))
    print(f"Episode {i + 1} complete with reward {episode_reward} and {len(episode_observation)} steps")
    env.stop_all_agents()
    return episode_observation


def save_to_npz(data: Dict[str, List[Any]], i: int) -> None:
    episode_length: int = len(data["reward"])
    episode_reward: float = sum(data["reward"])
    file_name: str = f"episode_{i}_{episode_length}_{episode_reward}.npz"

    rand_int: int = np.random.randint(0, 10)
    if rand_int != 1 and rand_int != 2:
        file_path: str = os.path.join(f"{DATASET_DIR}/train", file_name)
        print(f"Episode {i} saved as training data")
    else:
        file_path: str = os.path.join(f"{DATASET_DIR}/eval", file_name)
        print(f"Episode {i} saved as evaluation data")
    np.savez(file_path, **data)


def log_data(data: Dict[str, List[Any]], i: int) -> None:
    episode_length: int = len(data["reward"])
    episode_reward: float = sum(data["reward"])
    wandb.log(data={
        "reward": episode_reward,
        "length": episode_length
    }, step=i)


def run_generator(raster_queue: Queue, data_queue: Queue) -> None:
    env: OnlineQLabEnv = prepare_cross_road_env()
    for i in range(1000):
        try:
            ego_episode_observation: List[Dict[str, Any]] = run_episode(env, i, raster_queue)
            episode_data: Dict[str, List[Any]] = transform_observation(ego_episode_observation)
            # save_to_npz(episode_data, i)
            # log_data(episode_data, i)
            data_queue.put(episode_data)

            if data_queue.qsize() > 100: # Remove this code segment when the trainer is ready
                raise NotImplementedError("Please implement a method to get the data from the queue in the trainer")
        except AnomalousEpisodeException:
            print("Anomalous episode detected, skipping...")
        finally:
            time.sleep(2)
