import os
import time
import logging
from datetime import datetime
from collections import defaultdict

import mlflow
import torch
import numpy as np
from torch.cuda.amp import GradScaler

from core.utils.tools import configure_logging, mlflow_init, load_checkpoint, mlflow_log_metrics
from core.environment.wrappers import CollectionWrapper, ActionRewardResetWrapper
from td3.policy import TD3Agent
from constants import PREFILL, LOG_INTERVAL, SAVE_INTERVAL, MAX_TRAINING_STEPS, LOGBATCH_INTERVAL
from td3.exceptions import InsufficientDataException, StopTrainingException
import constants as C


class Trainer:
    def __init__(
        self,
        mlruns_dir: str,
        run_id: str,
        qcar_pos: list,
        waypoints: np.ndarray,
        device: str = C.cuda,
        prefill_steps: int = 0
    ) -> None:
        self.mlruns_dir: str = mlruns_dir
        self.run_id: str = run_id
        self.device: str = device
        self.prefill_steps: int = prefill_steps
        self.timer: float = time.time()
        self.last_backup_path: str = ''

    def setup_mlflow(self) -> tuple:
        configure_logging(prefix="[TRAIN]")
        # connect to the running mlflow instance
        os.environ["MLFLOW_RUN_ID"] = self.run_id
        mlrun = mlflow_init(self.mlruns_dir)
        # initialize mlflow
        mlflow.set_tracking_uri(self.mlruns_dir)
        # initialize data directory
        input_dir = self.mlruns_dir + f'/0/{self.run_id}/artifacts/episodes_train/0'
        eval_dir = self.mlruns_dir + f'/0/{self.run_id}/artifacts/episodes_eval/0'

        return input_dir, eval_dir

    def resume_training(self, agent: TD3Agent, resume: bool) -> None:
        if resume and len(self.data) >= PREFILL:
            logging.info(f'Resuming training from {self.run_id}')
            load_status: str = load_checkpoint(agent, self.mlruns_dir, self.run_id, map_location='cpu')
            logging.info(f'Trainer loaded model checkpoint status {load_status}')
            path_to_train_steps = f'{self.mlruns_dir[8:]}/0/{self.run_id}/metrics/train/steps'
            self.start_time = time.time()
            with open(path_to_train_steps) as f:
                last_line = f.readlines()[-1]
                self.steps = int(last_line.split()[2])
        else:
            self.start_time = time.time()
            self.steps = 0

    def resume_buffer(self) -> None:
        logging.info(f'Resuming buffer from {self.run_id}')
        self.data.reload_files()
        self.data.parse_and_load_buffer()
        self.data.last_load_time = time.time()
        logging.info(f'Buffer loaded with {len(self.data)} samples')

    def prepare_training(self, resume: bool = False) -> None: # setup function
        input_dir, eval_dir = self.setup_mlflow()
        torch.autograd.set_detect_anomaly(True)
        self.agent: TD3Agent = TD3Agent(input_dir)
        # whether we continue our training
        self.data = self.agent.buffer
        self.resume_buffer() # initial buffer load in case interrupted in prefill stage
        self.resume_training(agent=self.agent, resume=resume)
        # training parameters
        self.states = {}
        self.last_time: float = self.start_time
        self.last_steps: int = self.steps
        self.scaler: GradScaler = GradScaler(enabled=False)
        self.metrics: defaultdict = defaultdict(list)
        self.metrics_max: defaultdict = defaultdict(list)

    def update_agent_metrics(self, samples) -> None:
        metric_counter: int = 0
        if len(self.data) >= PREFILL:
            self.steps += 1
            actor_loss, critic_loss, gradients = self.agent.learn(samples)
            if actor_loss is not None and critic_loss is not None:
                self.metrics["actor_loss"] = actor_loss
                self.metrics["critic_loss"] = critic_loss
                if gradients["actor"] is not None and gradients["critic"] is not None:
                    self.metrics["actor_grad"] = gradients["actor"]
                    self.metrics["critic_grad"] = gradients["critic"]
                metric_counter += 1
                if metric_counter % 10 == 0:
                    print(self.metrics)
        else:
            raise InsufficientDataException()

    def log_training_metrics(self) -> None:
        if self.steps % LOG_INTERVAL != 0:
            return

        # cal average value and max value
        # self.metrics = {f'train/{k}': np.array(v.cpu()).mean() for k, v in self.metrics.items()}
        self.metrics = {f'train/{k}': np.array(v.cpu()).mean() if v is not None else None for k, v in
                        self.metrics.items()}
        self.metrics.update({f'train/{k}_max': np.array(v).max() for k, v in self.metrics_max.items()})
        self.metrics['train/steps'] = self.steps
        self.metrics['_step'] = self.steps
        self.metrics['_loss'] = self.metrics.get('train/loss_model', 0)
        self.metrics['_timestamp'] = datetime.now().timestamp()
        # cal fps
        time_stamp = time.time()
        time_diff = time_stamp - self.last_time
        if time_diff > 0:
            fps = (self.steps - self.last_steps) / time_diff
        else:
            fps = 0
        self.metrics['train/fps'] = fps
        # update time and steps
        self.last_time = time_stamp
        self.last_steps = self.steps
        if self.steps % 400 == 0 and self.steps > 0:
            logging.info(
                f"[steps{(self.steps - 400):06}]"
                f"  actor_loss: {self.metrics.get('train/actor_loss', 0):.3f}"
                f"  critic_loss: {self.metrics.get('train/critic_loss', 0):.3f}"
                f"  fps: {self.metrics.get('train/fps', 0):.3f}"
            )
            # skip first batch because the losses are very high and mess up y axis
            mlflow_log_metrics(self.metrics, step=self.steps-400)
        # clear metrics and metric_max
        self.metrics = defaultdict(list)
        self.metrics_max = defaultdict(list)

    def save_model(self, interrupt: bool = 'False') -> None:
        # skip if we do not have enough steps
        if self.steps % SAVE_INTERVAL != 0 or not interrupt:
            return

        # create checkpoint dict
        checkpoint = {}
        checkpoint["epoch"] = self.steps
        checkpoint["model_state_dict"] = self.agent.state_dict()
        checkpoint['optimizer_actor_state_dict'] = self.agent.actor_optimizer.state_dict()
        checkpoint['optimizer_critic_state_dict'] = self.agent.critic_optimizer.state_dict()

        checkpoint_path = f"{self.mlruns_dir[8:]}/0/{self.run_id}/latest_checkpoint.pt"
        backup_path = f"{self.mlruns_dir[8:]}/0/{self.run_id}/backup_{datetime.now().strftime('%Y%m%d%H%M%S')}.pt"
        # save model to disk
        try:
            torch.save(checkpoint, checkpoint_path)
            if time.time() - self.timer >= 600: # backup
                torch.save(checkpoint, backup_path)
                # os.remove(self.last_backup_path)
                # self.last_backup_path = backup_path
                self.timer = time.time()
            if self.steps % (SAVE_INTERVAL * 16) == 0:
                logging.info(f'Saved checkpoint {self.steps}')
        except IOError as e:
            logging.error(f"Failed to save checkpoint at {checkpoint_path}: {e}")

    def execute(self, interrupt: bool = True) -> None: # execution function
        samples = self.data.file_to_batch()
        self.update_agent_metrics(samples)
        self.log_training_metrics()
        self.save_model(interrupt)
        if self.steps >= MAX_TRAINING_STEPS:
            raise StopTrainingException()