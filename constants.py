print_QCar_info = False  # print "QCar configured successfully." in qcar.py or not
max_action = 0.5
discount = 0.99
state_dim = 10
action_dim = 2
lr = 2e-4  # 3e-4
tau = 0.004  # 0.005
observation_shape = (10,)
buffer_update_rate = 2
policy_freq = 4  # 2
# generator constants
METRIC_PREFIX: str = 'agent'
COOL_DOWN_TIME: float = 3.0
# trainer constants
MAX_TRAINING_STEPS = 100_000_000000
SAVE_INTERVAL = 30
LOGBATCH_INTERVAL = 400
LOG_INTERVAL = 10
QCAR_POS = []
# env constants
MAX_LOOKAHEAD_INDICES: int = 200
GOAL_THRESHOLD: int = 0.05


# settings
DEFAULT_MAX_STEPS: int = 400
batch_size = 1024
PREFILL = 20000
# batch_size = 2
# PREFILL = 2
# RUN_ID = '6df5a268e02e411ba7913d45e5396a8a'
RUN_ID = ''
action_v = 0.08
cuda = "cuda:1"
# start_point = 1700