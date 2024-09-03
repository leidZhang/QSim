import os

import torch

MODEL_TYPE: str = "Reinformer"
DATASET_DIR: str = "assets"
CONTEXT_LEN: int = 10
N_BLOCKS: int = 6
EMBED_DIM: int = 512  # 21174
N_HEADS: int = 2
DROPOUT_P: float = 0.1
GRAD_NORM: float = 0.25
TAU: float = 0.99
BATCH_SIZE: int = 1 # 64
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
RESUME: bool = True

# Set environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'