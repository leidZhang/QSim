print_QCar_info: bool = False  # print "QCar configured successfully." in qcar.py or not
max_action: float = 0.5
discount: float = 0.99
state_dim: int = 10
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
PREFILL: int = 100000
# RUN_ID = '1edfead4d1bc4c2ca91f3102741027d1'
RUN_ID: str = ''
action_v: float = 0.08
cuda: str = "cuda:1"
# start_point = 1700