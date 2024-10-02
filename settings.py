import os
from typing import List

import torch
import numpy as np

DATASET_DIR: str = "assets"
MODEL_TYPE: str = "Reinformer"
DATASET_DIR: str = "assets"
CONTEXT_LEN: int = 4
N_BLOCKS: int = 4
EMBED_DIM: int = 512  # 21174
N_HEADS: int = 2
DROPOUT_P: float = 0.1
GRAD_NORM: float = 0.25
TAU: float = 0.99
BATCH_SIZE: int = 32 # 64
LR: float = 1e-4
WD: float = 1e-4
WARMUP_STEPS: int = 5000
MAX_TRAIN_ITERS: int = 10
NUM_UPDATES_PER_ITER: int = 5000
DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED: int = 2024
INIT_TEMPERATURE: float = 0.1
USE_WANDB: bool = True
STATE_DIM: int = 6  # 400
ACT_DIM: int = 2
MODEL_PATH: str = os.path.join(os.getcwd(), "Reinformer/models/latest_checkpoint.pt")
RESUME: bool = False
WORKERS: int = 4
DEVICE: str = "cuda:1"
SEED: int = 2024
INIT_TEMPERATURE: float = 0.1

# Set environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
REFERENCE_POSE: List[float] = [0.15, 0.950, np.pi, 0.0, 0.0, 0.0]
IMAGE_SIZE: tuple = (84, 84)

# HITL settings
IP: str = "127.0.0.1"
HITL_PORT: int = 5000

print_QCar_info: bool = False  # print "QCar configured successfully." in qcar.py or not
max_action: float = 0.5
discount: float = 0.99
state_dim: int = 400
action_dim: int = 2
lr: float = 1e-4  # 3e-4
tau: float = 0.005
observation_shape: tuple = (10,)
buffer_update_rate: int = 2
policy_freq: int = 10  # 2
# generator constants
METRIC_PREFIX: str = 'agent'
COOL_DOWN_TIME: float = 3.0
# trainer constants
MAX_TRAINING_STEPS: int = 100_000_000000
SAVE_INTERVAL: int = 30
LOGBATCH_INTERVAL: int = 400
LOG_INTERVAL: int = 10
QCAR_POS: list = []
# env constants
MAX_LOOKAHEAD_INDICES: int = 200
GOAL_THRESHOLD: int = 0.05
RECOVER_INDICES: list = [0, 6, 8]

# settings
DEFAULT_MAX_STEPS: int = 2000
# batch_size = 2048
# PREFILL = 10000
batch_size: int = 2048
PREFILL: int = 3_400_000
RUN_ID = 'e4eef53e8c3a49a0b2967fa6be338fd2'
# RUN_ID: str = ''
action_v: float = 0.08
cuda: str = "cuda:1"
# start_point = 1700