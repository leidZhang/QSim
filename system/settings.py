import os
from typing import List

# training settings
EGO_VEHICLE_TASK: List[int] = [10, 2, 4, 6, 8, 10]
BOT_TASKS: List[List[int]] = [
    [23, 21, 16, 18, 11, 12, 8],
    [1, 13, 19, 17, 15, 5, 3, 1],
]
DT: float = 0.005
COLLISION_PENALTY: float = -40

# network settings
IP: str = '127.0.0.1'
PORTS: List[int] = [5000, 5001, 5002] # acotr ports
ENV_PORT: int = 8080
HITL_PORT: int = 8081
SUCCESS_CODE: int = 200
FAIL_CODE: int = 400

# file settings
PROJECT_DIR: str = os.getcwd()
NPZ_DIR: str = os.path.join(PROJECT_DIR, 'assets/npz')
WEIGHT_DIR: str = os.path.join(PROJECT_DIR, 'assets/model')
WEIGHT_FILENAME: str = 'last_checkpoint.pt'