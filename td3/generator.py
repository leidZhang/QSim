import time
import logging
from datetime import datetime
from collections import defaultdict

import numpy as np
from mlflow import set_tracking_uri

from core.policies.pure_persuit import PurePursuitPolicy
from core.policies.network import NetworkPolicy
from core.utils.tools import configure_logging, LogColorFormatter, load_checkpoint
from core.utils.tools import mlflow_log_metrics
from core.environment.wrappers import CollectionWrapper, ActionRewardResetWrapper
from core.environment.environment import QLabEnvironment
from core.environment import AnomalousEpisodeException
from core.data.data_TD3 import MlflowDataRepository, MlflowEpisodeRepository

from .policy import TD3Agent
from .environment import WaypointEnvironment
from constants import PREFILL, METRIC_PREFIX, COOL_DOWN_TIME


class Generator:
    def __init__(
        self,
        mlruns_dir: str,
        train_repo: str,
        eval_repo: str,
        qcar_pos: list,
        waypoints: np.ndarray,
        privileged: bool = True
    ) -> None:
        self.episode_num: int = 10000
        self.metrics_agg = defaultdict(list)
        self.mlruns_dir: str = mlruns_dir
        base_env: QLabEnvironment = WaypointEnvironment(dt=0.02, privileged=privileged)
        self.env: CollectionWrapper = CollectionWrapper(ActionRewardResetWrapper(base_env, qcar_pos, waypoints))
        self.train_repository: MlflowEpisodeRepository = MlflowEpisodeRepository(train_repo)
        self.eval_repository: MlflowEpisodeRepository = MlflowEpisodeRepository(eval_repo)
        set_tracking_uri(self.mlruns_dir)
        configure_logging(prefix='[GENERATOR]', info_color=LogColorFormatter.GREEN)

    def prepare_session(self, run_id: str, resume: bool, saved_data: int) -> tuple:
        steps, episodes = 0, 0
        if resume:
            _, steps, episodes = self.train_repository.count_steps()
            if saved_data >= PREFILL:
                train_repo: str = self.train_repository.artifact_uris
                self.policy = TD3Agent(train_repo)
                status: bool = load_checkpoint(self.policy, self.mlruns_dir, run_id, map_location='cpu')
                logging.info(f"Generator model checkpoint load status: {status}")
            else: 
                self.policy = PurePursuitPolicy(max_lookahead_distance=0.5)
        else:
            self.policy = PurePursuitPolicy(max_lookahead_distance=0.5)

        self.last_load_time = time.perf_counter()
        return steps, episodes

    def load_policy(self, run_id: str, saved_data: int) -> None:
        # print(saved_data)
        if type(self.policy) is not TD3Agent and saved_data >= PREFILL:
            logging.info("Prefill Complete, switching to main policy")
            train_repo: str = self.train_repository.artifact_uris
            self.policy = TD3Agent(train_repo)
            # is_prefill_policy = False

        if type(self.policy) is TD3Agent and time.perf_counter() - self.last_load_time > 30:
            model_step = load_checkpoint(self.policy, self.mlruns_dir, run_id)
            while model_step is None:
                model_step = load_checkpoint(self.policy, self.mlruns_dir, run_id)
                logging.debug('Generator model checkpoint not found, waiting...')
                time.sleep(10)
            logging.info(f'Generator loaded model checkpoint {model_step}')
            self.last_load_time = time.perf_counter()

    def run_episode(self, episdoe_steps: int, steps: int) -> tuple:
        metrics = defaultdict(list)
        observation, reward, done, info = self.env.reset()
        while not done:
            if type(self.policy) is TD3Agent:
                '''
                # event driven architecture for keyboard pvp?
                human interrupt : bool
                if interapute:
                    action = human_action()
                else:
                    action, metric = self.policy.select_action(observation['state'])
                '''
                action, metric = self.policy.select_action(observation['state'], steps)

                # filtered action = human and agent
                next_observation, reward, done, info = self.env.step(action, metric)
                self.policy.store_transition(observation['state'], action, reward, next_observation['state'], done)
                observation = next_observation
            else:
                action, metric = self.policy(observation)
                observation, reward, done, info = self.env.step(action, metric)

            for key, val in metric.items():
                metrics[key].append(val)

            episdoe_steps += 1
            steps += 1  
        self.env.step(np.zeros(2), {})  # stop the car
        if type(self.policy) is TD3Agent:
            time.sleep(COOL_DOWN_TIME)
        # print(metrics)
        return info, episdoe_steps, steps, metrics

    def log_episode(self, data: dict, episode_steps: int, steps: int, episodes: int, saved_data, metrics: dict) -> dict:
        logging.info(
            f"Episode recorded:"
            f"  steps: {episode_steps}"
            f",  reward: {data['reward'].sum():.1f}"
            f",  total steps: {steps:.0f}"
            f",  episodes: {episodes}"
        )
        # log metrics on mlflow
        metrics = {f'{METRIC_PREFIX}/{k}': np.array(v).mean() for k, v in metrics.items()}
        metrics.update({
            f'{METRIC_PREFIX}/episode_length': episode_steps,
            f'{METRIC_PREFIX}/steps': steps,  # All steps since previous restart
            f'{METRIC_PREFIX}/data_saved': saved_data,  # Steps saved in the training repo
            f'{METRIC_PREFIX}/episodes': episodes,
            f'{METRIC_PREFIX}/return': data['reward'].sum()
        })
        return metrics

    def aggregate_metrics(self, metrics: defaultdict, episodes:int, run_id:str) -> None:
        for key, val in metrics.items():
            self.metrics_agg[key].append(val)

        if len(self.metrics_agg[f'{METRIC_PREFIX}/return']) >= 1: # log_every:
            metrics_agg_max = {k: np.array(v).max() for k, v in self.metrics_agg.items()}
            self.metrics_agg = {k: np.array(v).mean() for k, v in self.metrics_agg.items()}
            self.metrics_agg[f'{METRIC_PREFIX}/return_max'] = metrics_agg_max[f'{METRIC_PREFIX}/return']
            self.metrics_agg['_timestamp'] = datetime.now().timestamp()
            # log metrics
            mlflow_log_metrics(self.metrics_agg, step=episodes, run_id=run_id)  # use episode number as step
            self.metrics_agg = defaultdict(list)

    def save_to_replay_buffer(self, data: dict, datas: list, episodes: int) -> int:
        accumulator = 0
        data_episodes = len(data)
        datas_steps = len(data['reset']) - 1
        # print(f"Current data step: {datas_steps}")

        # save data as npz
        if np.random.rand() < 0: # 0.2
            self.eval_repository.save_data(data, episodes - data_episodes, episodes - 1, 0)
        else:
            self.train_repository.save_data(data, episodes - data_episodes, episodes - 1, 0)
            accumulator = datas_steps

        return accumulator

    def execute(self, run_id: str, resume: bool) -> tuple:
        datas: list = []
        _, saved_data, _ = self.train_repository.count_steps()
        steps, episodes = self.prepare_session(run_id=run_id, resume=resume, saved_data=saved_data)

        for _ in range(self.episode_num):
            try:
                logging.info("Starting new episode...")
                self.load_policy(run_id, saved_data) # problem here
                episode_steps: int = 0
                info, episode_steps, steps, metrics = self.run_episode(episode_steps, steps)
                episodes += 1

                data = info["episode"]
                metrics = self.log_episode(data, episode_steps, steps, episodes, saved_data, metrics)
                self.aggregate_metrics(metrics, episodes, run_id) # aggregate metrics
                saved_data += self.save_to_replay_buffer(data, datas, episodes)
            except AnomalousEpisodeException as e:
                print(e)
            # except Exception as e:
            #     print(e)
