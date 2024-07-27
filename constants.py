print_QCar_info: bool = False  # print "QCar configured successfully." in qcar.py or not
max_action: float = 0.5
discount: float = 0.99
state_info_dim: int = 6
action_dim: int = 2
lr: float = 3e-4  # 3e-4
tau: float = 0.005
resolution: tuple = (84, 84)
buffer_update_rate: int = 2
policy_freq: int = 2  # 2
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
PREFILL = 5000
batch_size: int = 512
# PREFILL: int = 3_400_000
RUN_ID = ''
# RUN_ID = '9bc59b35e44844a6a4ba79993f615ea1'
# RUN_ID: str = '9e2db53711c744888cdb3140ef44536c'
action_v: float = 0.08
cuda: str = "cuda:1"
# start_point = 1700