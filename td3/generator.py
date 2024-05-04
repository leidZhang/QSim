import time
import logging
import datetime
from collections import defaultdict

import numpy as np
from mlflow import set_tracking_uri

from core.policies.pure_persuit import PurePursuitPolicy
from core.policies.network import NetworkPolicy
from core.utils.tools import configure_logging, LogColorFormatter, load_checkpoint
from core.utils.tools import mlflow_log_metrics
from core.environment.wrappers import CollectionWrapper, ActionRewardResetWrapper
from core.environment.primary import QLabEnvironment
from core.data.data_TD3 import MlflowDataRepository, MlflowEpisodeRepository

from .policy import TD3Agent
from . import constants as C


class Generator:
    def __init__(self, mlruns_dir: str, train_repo: str, eval_repo: str, privileged: bool = True) -> None:
        self.episode_num: int = 10000
        self.mlruns_dir: str = mlruns_dir
        self.env: CollectionWrapper = CollectionWrapper(QLabEnvironment(dt=0.05, privileged=privileged))
        self.train_repository: MlflowEpisodeRepository = MlflowEpisodeRepository(train_repo)
        self.eval_repository: MlflowEpisodeRepository = MlflowEpisodeRepository(eval_repo)
        set_tracking_uri(self.mlruns_dir)
        configure_logging(prefix='[GENERATOR]', info_color=LogColorFormatter.GREEN)

    def prepare_training_session(self, run_id: str) -> None:
        steps, episodes = 0, 0

        if run_id != '':
            self.policy = TD3Agent(self.env, self.train_repository)
            status: bool = load_checkpoint(self.policy, self.mlruns_dir, run_id, map_location='cpu')
            logging.info(f"Generator model checkpoint load status: {status}")
            _, steps, episodes = self.train_repository.count_steps()
        else:
            self.policy = PurePursuitPolicy(max_lookahead_distance=0.5)

        self.last_load_time = time.perf_counter()
        return steps, episodes

    def load_policy(self, is_prefill_policy: bool, saved_data: int) -> None:
        if is_prefill_policy and saved_data >= C.prefill:
            logging.info("Prefill Complete, switching to main policy")
            train_repo: str = self.train_repository.artifact_uris
            self.policy = TD3Agent(self.env, train_repo)
            is_prefill_policy = False

        if not is_prefill_policy and self.last_load_time > 30:
            model_step = load_checkpoint(self.policy, self.mlruns_dir)
            while model_step is None:
                model_step = load_checkpoint(self.policy, self.mlruns_dir)
                logging.debug('Generator model checkpoint not found, waiting...')
                time.sleep(10)
            logging.info(f'Generator loaded model checkpoint {model_step}')
            self.last_load_time = time.perf_counter()

    def run_episode(self, is_prefill_policy: bool, episdoe_steps: int, steps: int) -> tuple:
        metrics = defaultdict(list)
        observation, reward, done, info = self.env.reset()
        while not done:
            if not is_prefill_policy:
                action, metric = self.policy.select_action(observation['state'])
                next_observation, reward, done, info = self.env.step(action, metrics)
                observation = next_observation
            else:
                action, metric = self.policy(observation)
                observation, reward, done, info = self.env.step(action, metrics)

            for key, val in metric.items():
                metrics[key].append(val)

            episdoe_steps += 1
            steps += 1 # ???
        return info, episdoe_steps, steps, metrics

    def log_metrics(self, data: dict, episode_steps: int, steps: int, episodes: int, saved_data, metrics: dict) -> None:
        logging.info(
            f"Episode recorded:"
            f"  steps: {episode_steps}"
            f",  reward: {data['reward'].sum():.1f}"
            f",  total steps: {steps:.0f}"
            f",  episodes: {episodes}"
        )
        # log metrics on mlflow
        metrics = {f'{C.METRIC_PREFIX}/{k}': np.array(v).mean() for k, v in metrics.items()}
        metrics.update({
            f'{C.METRIC_PREFIX}/episode_length': episode_steps,
            f'{C.METRIC_PREFIX}/steps': steps,  # All steps since previous restart
            f'{C.METRIC_PREFIX}/data_saved': saved_data,  # Steps saved in the training repo
            f'{C.METRIC_PREFIX}/episodes': episodes,
            f'{C.METRIC_PREFIX}/return': data['reward'].sum()
        })

    def aggregate_metrics(self, metrics: dict, episodes:int) -> None:
        metrics_agg: defaultdict = defaultdict(list)
        for key, val in metrics.items():
            if not np.isnan(val):
                metrics_agg[key].append(val)
        if len(metrics_agg[f'{C.METRIC_PREFIX}/return']) >= 1: # log_every:
            metrics_agg_max = {k: np.array(v).max() for k, v in metrics_agg.items()}
            metrics_agg = {k: np.array(v).mean() for k, v in metrics_agg.items()}
            metrics_agg[f'{C.METRIC_PREFIX}/return_max'] = metrics_agg_max[f'{C.METRIC_PREFIX}/return']
            metrics_agg['_timestamp'] = datetime.now().timestamp()

            mlflow_log_metrics(metrics_agg, step=episodes)  # use episode number as step
            metrics_agg = defaultdict(list)

    def save_to_replay_buffer(self, data: dict, datas: list, episodes: int) -> int:
        accumulator = 0
        datas.append(data)
        data_episodes = len(data)
        datas_steps = sum(len(d['reset']) - 1 for d in datas)
        # print(f"Current data step: {datas_steps}")
        if datas_steps >= 0: # steps_per_npz
            # concatenate episodes
            data = {}
            for key in datas[0]:
                data[key] = np.concatenate([b[key] for b in datas], axis=0)
            datas = []

            # save data as npz
            if np.random.rand() < 0.2:
                self.eval_repository.save_data(data, episodes - data_episodes, episodes - 1, 0)
            else:
                self.train_repository.save_data(data, episodes - data_episodes, episodes - 1, 0)
                accumulator = datas_steps

        return accumulator

    def __call__(self, run_id: str) -> tuple:
        datas = []
        steps, episodes = self.prepare_training_session(run_id=run_id)
        _, saved_data, _ = self.train_repository.count_steps()
        is_prefill_policy: bool = run_id == ''

        for _ in range(self.episode_num):
            self.load_policy(is_prefill_policy, saved_data)

            episode_steps: int = 0
            if isinstance(self.policy, NetworkPolicy):
                self.policy.reset_state()
            info, episode_steps, steps, metrics = self.run_episode(is_prefill_policy, episode_steps, steps)
            episodes += 1

            data = info["episode"]
            self.log_metrics(data, episode_steps, steps, episodes, saved_data, metrics)
            self.aggregate_metrics(metrics, episodes) # aggregate metrics
            saved_data += self.save_to_replay_buffer(data, datas, episodes)
