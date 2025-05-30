import cv2
import wandb
import numpy as np

import csv
import time
from copy import deepcopy
from queue import Queue
from datetime import datetime
from typing import Dict, List, Tuple, Any

from qvl.qlabs import QuanserInteractiveLabs
from core.roadmap import ACCRoadMap
from core.environment import AnomalousEpisodeException
from core.environment.builder import GeneralMapBuilder
from core.environment.director import GeneralDirector
from core.environment.constants import FULL_CONFIG
from core.environment.simulator import QLabSimulator
from core.policies import PurePursuiteAdaptor
from settings import *
from reinformer import ReinformerPolicy, ReinFormer
from environment import CrossRoadEnvironment, OnlineQLabEnv

def prepare_reinformer_policy() -> ReinformerPolicy:
    model: ReinFormer = ReinFormer(
        state_dim=STATE_DIM,
        act_dim=ACT_DIM,
        n_blocks=N_BLOCKS,
        h_dim=EMBED_DIM,
        context_len=CONTEXT_LEN,
        n_heads=N_HEADS,
        drop_p=DROPOUT_P,
        init_temperature=INIT_TEMPERATURE,
        target_entropy=-ACT_DIM
    ).to(DEVICE)
    policy: ReinformerPolicy = ReinformerPolicy(model, weight_path="latest_checkpoint.pt")
    return policy


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
    env: OnlineQLabEnv = CrossRoadEnvironment(sim, roadmap, privileged=True, dt=0.1)
    # policy: ReinformerPolicy = prepare_reinformer_policy()
    # env.set_ego_policy(policy)
    env.set_ego_policy(PurePursuiteAdaptor(max_lookahead_distance=0.68))
    # env.set_ego_policy(CompositeDQNPolicy())
    env.set_hazard_policy(PurePursuiteAdaptor())

    return env


# TODO: Transform data
def transform_observation(episode_observation: List[Dict[str, Any]]) -> Dict[str, list]:
    data: Dict[str, dict] = {
        'state_info': [], 
        'waypoints': [],
        'violation': []
    }

    for step_observation in episode_observation:
        data['state_info'].append(step_observation['state_info'])
        data['waypoints'].append(step_observation['waypoints'])
        data['violation'].append(step_observation['violation'])

    return data


def run_episode(env: CrossRoadEnvironment, i: int) -> Tuple[list, dict]:
    # print(f"Starting episode {i}...")
    episode_reward, episode_observation, info = 0, [], {}
    # use this to set the agents to their initial positions
    observation, reward, done, _ = env.reset()
    for i in range(300):
        start: float = time.perf_counter()
        observation, reward, done, info = env.step()
        info["step"] = i
        episode_observation.append(observation)
        episode_reward += reward
        if done:
            break
        elapsed_time: float = time.perf_counter() - start
        time.sleep(max(0, env.dt - elapsed_time))
        # print("Step time elapsed: ", elapsed_time)
    # print(f"Episode {i + 1} complete with reward {episode_reward} and {len(episode_observation)} steps")

    return episode_observation, info


def save_to_npz(data: Dict[str, List[Any]], i: int) -> None:
    timestamp: str = datetime.now().strftime("%Y%m%d")
    file_name: str = f"episode_{i}_{timestamp}.npz"

    rand_int: int = np.random.randint(0, 10)
    if rand_int != 1 and rand_int != 2:
        file_path: str = os.path.join(f"{DATASET_DIR}/train", file_name)
        print(f"Episode {i} saved as training data")
    else:
        file_path: str = os.path.join(f"{DATASET_DIR}/eval", file_name)
        print(f"Episode {i} saved as evaluation data")
    np.savez(file_path, **data)


# Save to png and csv
def save_data(data: Dict[int, dict]) -> None:
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Timestamp: ", timestamp)

    for i in range(4):
        folder_path: str = f"{DATASET_DIR}/{timestamp}/{i}"
        os.makedirs(folder_path, exist_ok=True)
        csv_path: str = os.path.join(folder_path, 'state_info.csv')
        for j in range(len(data[i]['raster_map'])):
            cv2.imwrite(f"{folder_path}/{j}.png", data[i]['raster_map'][j])
        
        with open(csv_path, "w", newline="", encoding='utf-8') as f:
            headers = ['x', 'y', 'yaw', 'v', 'a']
            state_info = data[i]['state_info']  
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(state_info)

def log_data(data: Dict[str, List[Any]], i: int) -> None:
    episode_length: int = len(data["reward"])
    episode_reward: float = sum(data["reward"])
    wandb.log(data={
        "reward": episode_reward,
        "length": episode_length
    }, step=i)

def run_generator() -> None:
    env: OnlineQLabEnv = prepare_cross_road_env()
    for i in range(10000):
        try:
            print(f"Starting episode {i}")
            episode_observation, info = run_episode(env, i)
            episode_data: Dict[str, List[Any]] = transform_observation(episode_observation)
            # save_to_npz(episode_data, i)                
            print(f"Saved the episode {i} with steps", info['step'])

            # save_to_npz(episode_data, i)
            # log_data(episode_data, i)
            # data_queue.put(episode_data)
        except AnomalousEpisodeException:
            print("Anomalous episode detected, skipping...")
            i -= 1
        finally:
            env.stop_all_agents()
            time.sleep(1)
